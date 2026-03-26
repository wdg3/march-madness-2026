"""PBP Deep Matchup Model — Hierarchical Play → Game → Season → Prediction.

Every play from a team's season, conditioned on every player on the court,
feeds into the matchup prediction.  No aggregation to team-level features.

Architecture
    PlayEncoder:    10 player embeddings + play context → play embedding
    GameGRU:        GRU over play sequence → game embedding (captures trajectory)
    SeasonEncoder:  Causal attention over game sequence → season embedding
    MatchupHead:    Two season embeddings → P(Team A wins) + margin
"""

import torch
import torch.nn as nn

# Context features per play (no score — prevents leakage during game pretraining)
# [period_frac, time_frac, is_scoring, score_value_norm, is_shooting, our_action]
N_PLAY_CONTEXT = 6


class PlayEncoder(nn.Module):
    """Encode a single play from one team's perspective.

    Input per play:
        our_players   (5,) int  — indices of our 5 players on court
        their_players (5,) int  — indices of opponent's 5 players
        play_type     ()  int   — play-type index
        context       (6,) float — see N_PLAY_CONTEXT

    All 10 players go through self-attention so the model can learn:
      - Lineup chemistry (how teammates interact)
      - Matchup dynamics (how our players match up against theirs)
    A learned side embedding distinguishes our team from theirs.
    """

    def __init__(self, n_players, player_dim=32, n_play_types=25,
                 ptype_dim=8, out_dim=64, dropout=0.1):
        super().__init__()
        self.player_emb = nn.Embedding(n_players + 1, player_dim, padding_idx=0)
        self.side_emb = nn.Embedding(2, player_dim)  # 0=their, 1=our
        n_attn_heads = max(1, player_dim // 4)  # 4 dims per head minimum
        self.lineup_attn = nn.MultiheadAttention(
            player_dim, num_heads=n_attn_heads, batch_first=True, dropout=dropout,
        )
        self.lineup_norm = nn.LayerNorm(player_dim)
        self.ptype_emb = nn.Embedding(n_play_types + 1, ptype_dim, padding_idx=0)

        in_dim = 2 * player_dim + ptype_dim + N_PLAY_CONTEXT
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, our, their, ptypes, ctx):
        """All args have leading shape (*, ...).  Returns (*, out_dim)."""
        our_pe = self.player_emb(our)        # (*, 5, player_dim)
        their_pe = self.player_emb(their)    # (*, 5, player_dim)

        # Add side embeddings so attention knows who's on which team
        all_p = torch.cat([
            our_pe + self.side_emb.weight[1],    # our side
            their_pe + self.side_emb.weight[0],  # their side
        ], dim=-2)                               # (*, 10, player_dim)

        # Self-attention over all 10 players (lineup chemistry + matchup dynamics)
        orig_shape = all_p.shape
        flat = all_p.reshape(-1, 10, orig_shape[-1])  # (N, 10, player_dim)
        attn_out, _ = self.lineup_attn(flat, flat, flat)
        attn_out = self.lineup_norm(flat + attn_out)   # residual + norm
        attn_out = attn_out.reshape(orig_shape)

        # Split back into our/their and pool
        our_e = attn_out[..., :5, :].mean(dim=-2)     # (*, player_dim)
        their_e = attn_out[..., 5:, :].mean(dim=-2)

        pt_e = self.ptype_emb(ptypes)                   # (*, ptype_dim)
        x = torch.cat([our_e, their_e, pt_e, ctx], dim=-1)
        return self.norm(self.mlp(x))


class SeasonEncoder(nn.Module):
    """Causal transformer over a sequence of game embeddings.

    Games can only attend to *earlier* games — strict causal mask.
    Output is the last-token representation (most-recent game's output),
    which encodes the team's full trajectory up to that point.
    """

    def __init__(self, dim=64, n_heads=4, n_layers=2, dropout=0.1,
                 max_games=50):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_games, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, game_embs, mask=None):
        """
        game_embs : (B, G, D)
        mask      : (B, G) bool — True = padding
        Returns   : (B, D)
        """
        B, G, D = game_embs.shape
        x = game_embs + self.pos[:, :G, :]

        causal = torch.triu(
            torch.ones(G, G, device=x.device), diagonal=1
        ).bool()

        out = self.norm(self.enc(x, mask=causal, src_key_padding_mask=mask))

        # last non-padding position
        if mask is not None:
            lengths = (~mask).sum(dim=1).clamp(min=1) - 1
            return out[torch.arange(B, device=out.device), lengths]
        return out[:, -1, :]


class PBPMatchupModel(nn.Module):
    """End-to-end PBP matchup predictor.

    Full gradient path:
        plays → PlayEncoder → GRU → game_emb → game_proj → SeasonEncoder
        → season_emb → MatchupHead → P(win)

    Auxiliary game-level loss reuses the same heads for dense play encoder
    supervision without a separate training phase.
    """

    def __init__(self, n_players, embed_dim=64, player_dim=32,
                 n_play_types=25, ptype_dim=8,
                 n_heads=4, n_season_layers=2, dropout=0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.player_dim = player_dim

        self.play_encoder = PlayEncoder(
            n_players, player_dim, n_play_types, ptype_dim,
            embed_dim, dropout,
        )

        # GRU over play sequence — captures game trajectory, momentum, clutch plays
        self.game_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.game_norm = nn.LayerNorm(embed_dim)

        self.season_encoder = SeasonEncoder(
            embed_dim, n_heads, n_season_layers, dropout,
        )

        # Project enriched game context:
        # [our_game_emb, opp_game_emb, won, margin] → embed_dim
        self.game_proj = nn.Sequential(
            nn.Linear(2 * embed_dim + 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Shared matchup heads (game-level aux + season-level main)
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.margin_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    # ── building blocks ────────────────────────────────────────────

    def encode_game(self, our, their, ptypes, ctx, play_mask=None):
        """Encode one team's view of a game → game embedding.

        Plays are processed in time order through a GRU, so the game
        embedding captures trajectory/momentum, not just play counts.

        Args:
            our        : (B, T, 5) int
            their      : (B, T, 5) int
            ptypes     : (B, T) int
            ctx        : (B, T, 6) float
            play_mask  : (B, T) bool — True = padding
        Returns:
            game_emb   : (B, D)
        """
        pe = self.play_encoder(our, their, ptypes, ctx)  # (B, T, D)

        if play_mask is not None:
            lengths = (~play_mask).sum(dim=-1).clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                pe, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            _, h = self.game_gru(packed)  # h: (1, B, D)
        else:
            _, h = self.game_gru(pe)      # h: (1, B, D)

        return self.game_norm(h.squeeze(0))

    def predict_from_game_embs(self, emb_a, emb_b):
        """Predict matchup from two game embeddings (phase 1)."""
        x = torch.cat([emb_a, emb_b], dim=-1)
        return self.head(x).squeeze(-1), self.margin_head(x).squeeze(-1)

    def encode_season_enriched(self, our_embs, opp_embs, outcomes, mask=None):
        """Encode season from enriched game representations.

        For each game in the season, we see both teams' game embeddings
        plus the outcome, giving the season encoder direct SOS signal.

        Args:
            our_embs:  (B, G, D) — our game embeddings
            opp_embs:  (B, G, D) — opponent game embeddings
            outcomes:  (B, G, 2) — [won, margin] per game
            mask:      (B, G) bool — True = padding
        Returns:
            season_emb: (B, D)
        """
        enriched = torch.cat([our_embs, opp_embs, outcomes], dim=-1)
        projected = self.game_proj(enriched)  # (B, G, D)
        return self.season_encoder(projected, mask)

    def predict(self, season_a, season_b):
        """Predict matchup from two season embeddings (phase 2 / inference)."""
        x = torch.cat([season_a, season_b], dim=-1)
        return self.head(x).squeeze(-1), self.margin_head(x).squeeze(-1)

    def matchup_embedding(self, season_a, season_b):
        """Extract the penultimate matchup embedding (64-dim).

        This is the hidden representation after the first linear + GELU
        in the matchup head — captures how two teams interact before
        collapsing to a scalar prediction.
        """
        x = torch.cat([season_a, season_b], dim=-1)
        # head[0] = Linear(128, 64), head[1] = GELU
        return self.head[1](self.head[0](x))
