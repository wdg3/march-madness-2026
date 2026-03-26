# Player Embeddings Design Document

## Goal

Add player-level signal to the March Madness prediction model so it can
detect roster-level changes (injuries, transfers, breakout players) that
team-aggregate features miss.  The market already prices these in; our model
currently cannot, creating adverse selection on our "edge" bets.

---

## Architecture Overview

```
                         DATA SOURCES
                         ============

  Barttorvik Advanced Stats          ncaahoopR Play-by-Play
  (74 cols, 2008-2026)               (lineup stints, 2002-2026)
  per-player per-season:              per-possession:
    BPM, PER, ORtg, DRtg,              which 5 players on court
    usage, shooting splits,             points scored/allowed
    win shares, height, pos,            per stint
    minutes, class, player_id
          |                                    |
          v                                    v
  ┌───────────────┐                  ┌──────────────────┐
  │  Player Stats │                  │  Compute RAPM    │
  │  (box-score   │                  │  (ridge regress  │
  │   features)   │                  │   on lineup      │
  │               │                  │   stints)        │
  └───────┬───────┘                  └────────┬─────────┘
          │                                   │
          v                                   v
  ┌─────────────────────────────────────────────────┐
  │              PLAYER EMBEDDING                   │
  │                                                 │
  │  Per player-season vector:                      │
  │    [RAPM_off, RAPM_def,                         │
  │     BPM, usage, ORtg, DRtg,                    │
  │     PER, win_shares,                            │
  │     height, position_enc,                       │
  │     class_year, minutes_pct,                    │
  │     recruit_rank (if avail)]                    │
  │                                                 │
  │  Keyed by: (player_id, season)                  │
  └─────────────────┬───────────────────────────────┘
                    │
                    v
  ┌─────────────────────────────────────────────────┐
  │         TEAM AGGREGATION LAYER                  │
  │                                                 │
  │  For each (TeamID, Season), select top-8        │
  │  players by minutes, then compute:              │
  │                                                 │
  │  1. Minutes-weighted means                      │
  │     mean_RAPM_off, mean_BPM, mean_usage, ...   │
  │                                                 │
  │  2. Star concentration                          │
  │     top1_min_pct, top3_min_pct                  │
  │     top1_BPM, max_usage                         │
  │                                                 │
  │  3. Depth/variance                              │
  │     std_BPM, std_RAPM                           │
  │     min_pct_player_8 (bench depth)              │
  │                                                 │
  │  4. Roster composition                          │
  │     n_upperclass_in_top5                        │
  │     avg_height_top5, position_balance            │
  │     n_new_players_in_top8 (portal/freshmen)     │
  │                                                 │
  │  5. Returning production                        │
  │     returning_BPM_sum (from prior year)         │
  │     lost_BPM_sum (departed players)             │
  │     portal_incoming_BPM_sum                     │
  │                                                 │
  │  Output: ~25-30 team-season features            │
  └─────────────────┬───────────────────────────────┘
                    │
                    │  New feature source: "player_impact"
                    │  plugs into existing registry
                    v
  ┌─────────────────────────────────────────────────┐
  │         EXISTING PIPELINE                       │
  │                                                 │
  │  build_team_features()                          │
  │    ├── massey (rankings)                        │
  │    ├── kenpom (efficiency)                      │
  │    ├── vegas (market lines)                     │
  │    ├── roster (continuity) ◄── keep, simpler    │
  │    ├── ... (18 other sources)                   │
  │    └── player_impact ◄── NEW                    │
  │           │                                     │
  │           v                                     │
  │  build_matchups()                               │
  │    creates _A, _B, _delta columns               │
  │    for all features including player_impact     │
  │           │                                     │
  │           v                                     │
  │  AutoGluon ensemble                             │
  │    LightGBM, XGBoost, CatBoost, RF, FastAI     │
  └─────────────────────────────────────────────────┘
```

---

## Data Join Strategy

```
  Barttorvik getadvstats CSV
  ┌──────────────────────────────────────────────────────┐
  │ player_id │ player_name │ team_name │ year │ ...74   │
  │  (int)    │  (str)      │  (str)    │(int) │  cols   │
  └─────┬────────────────────────┬───────────────────────┘
        │                        │
        │ player_id tracks       │ team_name needs
        │ across seasons         │ mapping to TeamID
        │ (transfers auto-       │
        │  handled: same         │
        │  player_id, new team)  │
        │                        │
        │                        v
        │              ┌───────────────────┐
        │              │ MTeamSpellings.csv│
        │              │ barttorvik name   │──► TeamID
        │              │ → TeamID mapping  │
        │              │ (already built in │
        │              │  roster.py)       │
        │              └───────────────────┘
        │
        v
  ┌─────────────────────────────────┐
  │  Player-Season Table            │
  │  (player_id, TeamID, Season)    │
  │                                 │
  │  Join to prior year by          │
  │  player_id to detect:           │
  │  - returning players            │
  │  - transfers (same player_id,   │
  │    different TeamID)            │
  │  - departed (in year N,         │
  │    not in year N+1 on team)     │
  └─────────────────────────────────┘


  ncaahoopR Play-by-Play CSVs
  ┌───────────────────────────────────────────────────┐
  │ game_id │ play_desc │ home_team │ away_team │ ... │
  │         │           │ home_1..5 │ away_1..5 │     │
  │         │           │ (player   │ (player   │     │
  │         │           │  names)   │  names)   │     │
  └─────┬─────────────────────────────────────────────┘
        │
        │ Need to join player names
        │ to barttorvik player_ids
        │ (fuzzy match on name + team)
        │
        v
  ┌─────────────────────────────────┐
  │  Lineup Stint Matrix            │
  │                                 │
  │  For each stint (continuous     │
  │  stretch with same 10 players   │
  │  on court):                     │
  │                                 │
  │  row = stint                    │
  │  cols = player indicators       │
  │    (+1 if on offense team,      │
  │     -1 if on defense team)      │
  │  target = points_per_100_poss   │
  │                                 │
  │  Ridge regression on this       │
  │  matrix → RAPM per player       │
  └─────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Expanded Box-Score Features (1-2 days)

Expand existing `roster.py` barttorvik fetch to capture all 74 columns.
Aggregate to team-level features. No new data sources needed.

**New features (~25):**
- `pi_mean_bpm`, `pi_top1_bpm`, `pi_std_bpm`
- `pi_mean_ortg`, `pi_mean_drtg`
- `pi_mean_usage`, `pi_max_usage`
- `pi_mean_per`, `pi_top1_per`
- `pi_top1_min_pct`, `pi_top3_min_pct` (star dependence)
- `pi_avg_height`, `pi_height_std`
- `pi_n_new_in_top8` (new players in rotation)
- `pi_returning_bpm_sum`, `pi_lost_bpm_sum` (production turnover)
- `pi_portal_incoming_bpm` (transfer talent)
- `pi_depth_8th_player_min_pct`

**Data flow:**
```
barttorvik 74-col CSV → player_stats_{year}.csv (cached)
    → aggregate by (TeamID, Season) → team features
    → plugs into build_team_features() as "player_impact"
```

### Phase 2: RAPM from Play-by-Play (1-2 weeks)

Download ncaahoopR play-by-play data. Parse lineup stints.
Compute regularized adjusted plus-minus per player-season.
Add RAPM to the player embedding vector.

**Challenges:**
- Player name matching between ncaahoopR and barttorvik
- Stint parsing from play-by-play events
- Ridge regression hyperparameter tuning (regularization strength)
- Computational cost (~24 seasons × ~5000 games/season)

### Phase 3: Availability-Adjusted Predictions (future)

For tournament predictions specifically, adjust the team aggregation
based on who is actually available (injury reports, suspensions).
This requires manual or semi-automated roster updates before each
tournament round.

**Mechanism:**
```
Default team features use full-season roster.
At prediction time, if a player is marked unavailable:
  - Recompute team aggregates excluding that player
  - Delta features capture the impact
  - Model sees a "weaker" version of the team
```

---

## Key Design Decisions

1. **Aggregate to team level, don't feed raw player vectors.**
   The matchup model expects (TeamA_features, TeamB_features, delta).
   Player embeddings are intermediate — the model sees team-level
   summaries derived from them. This avoids variable-length inputs
   and keeps the existing pipeline intact.

2. **Use player_id for tracking, not names.**
   Barttorvik's persistent player_id handles transfers automatically:
   same ID appears under a new team. This is critical for computing
   "returning production" and "portal incoming talent."

3. **Phase 1 is high-value, low-risk.**
   We already fetch from barttorvik. Expanding from 6 to 74 columns
   and computing smarter aggregates requires no new infrastructure.
   The UNC case would be directly addressed: their top player's BPM
   and minutes would show up in `pi_top1_bpm` and `pi_top1_min_pct`,
   and his absence would be capturable via availability adjustment.

4. **RAPM (Phase 2) adds the most novel signal.**
   Box-score stats are already partially captured by KenPom/Massey.
   RAPM measures player impact *controlling for teammates and
   opponents* — it's the closest thing to a true player value metric.
   This is what would let the model understand that losing Player X
   costs the team 3 points per 100 possessions, not just "a starter."
