"""Composable PBP Model — player-level representations from play-by-play data.

Teams are represented as compositions of available players. Each player's
representation is built entirely from play-by-play data: attention-pooling
over play embeddings from plays where that player was on court.

Players can be excluded at prediction time (injury, suspension) by masking
them out. The prediction is a direct function of individual player
interactions — no mean-pooling into team vectors.

Architecture
============
    PlayEncoder (shared, warm-started from PBP model)
        10 players + play context → play embedding (play_dim)
        Encodes all plays once per season, cached.

    Per-Player PBP Pooling
        For each rostered player, attention-pool over play embeddings
        where that player was on court.
        → d_player-dim per player

    Team Self-Attention
        Available players attend to teammates.
        Captures lineup chemistry / positional balance.

    Cross-Team Attention
        Team A's players attend to Team B's players.
        Captures matchup dynamics at the individual level.

    Prediction Attention
        A learned query attends over all matchup-aware player
        representations from both teams to produce the prediction.
        Every player's contribution flows to the output.
"""

import torch
import torch.nn as nn

from models.player_model import AttentionBlock

MAX_PLAYERS = 15  # NCAA roster limit — actual count varies per team
MAX_PLAYS_PER_PLAYER = 200


class PlayerPBPPooler(nn.Module):
    """Attention-weighted pooling of play embeddings for one player.

    A learned query attends over all play embeddings where the player
    was on court, selecting the most informative plays.
    """

    def __init__(self, play_dim: int, out_dim: int,
                 n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, play_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            play_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(play_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.default_emb = nn.Parameter(torch.randn(out_dim) * 0.02)

    def forward(self, play_embs, play_mask):
        """
        Args:
            play_embs:  (B, T, play_dim) — play embeddings where player was on court
            play_mask:  (B, T) bool — True = padding
        Returns:
            (B, out_dim)
        """
        B = play_embs.shape[0]
        all_masked = play_mask.all(dim=1)

        # Unmask first position for fully-masked players to avoid NaN
        safe_mask = play_mask.clone()
        safe_mask[all_masked, 0] = False

        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attn(
            query, play_embs, play_embs, key_padding_mask=safe_mask,
        )
        pooled = self.out_proj(attn_out.squeeze(1))

        # Replace cold-start output with learned default
        pooled[all_masked] = self.default_emb
        return pooled


class ComposablePBPModel(nn.Module):
    """Matchup predictor with composable player-level representations.

    No information is pooled into team-level vectors. Player identities
    are preserved through self-attention, cross-attention, and into the
    prediction head. Excluding a player changes every downstream
    computation because attention recomputes over the remaining roster.

    The prediction head uses a learned query that attends over all
    matchup-aware player representations from both teams, so every
    player's contribution directly influences the output.
    """

    def __init__(
        self,
        play_encoder: nn.Module,
        play_dim: int = 64,
        d_player: int = 64,
        n_heads: int = 4,
        n_self_layers: int = 2,
        n_cross_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.play_dim = play_dim
        self.d_player = d_player

        # Reuse trained PlayEncoder (warm-started, then fine-tuned)
        self.play_encoder = play_encoder

        # Per-player PBP pooling (play_dim → d_player)
        self.pbp_pooler = PlayerPBPPooler(
            play_dim, d_player, n_heads=2, dropout=dropout,
        )

        # Team self-attention (lineup chemistry)
        self.team_self_layers = nn.ModuleList([
            AttentionBlock(d_player, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

        # Cross-team attention (matchup dynamics at player level)
        self.cross_layers = nn.ModuleList([
            AttentionBlock(d_player, n_heads, dropout, is_cross=True)
            for _ in range(n_cross_layers)
        ])

        # Norms
        self.self_norm = nn.LayerNorm(d_player)
        self.cross_norm = nn.LayerNorm(d_player)

        # Side embedding: distinguish Team A players from Team B players
        # when concatenated for the prediction attention
        self.side_emb = nn.Embedding(2, d_player)

        # Prediction: learned query attends over all players from both teams
        self.pred_query = nn.Parameter(torch.randn(1, 1, d_player) * 0.02)
        self.pred_attn = nn.MultiheadAttention(
            d_player, n_heads, dropout=dropout, batch_first=True,
        )
        self.pred_norm = nn.LayerNorm(d_player)

        self.head = nn.Sequential(
            nn.Linear(d_player, d_player * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_player * 2, d_player),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_player, 1),
        )
        self.margin_head = nn.Linear(d_player, 1)

    def encode_players(self, player_play_embs, player_play_masks):
        """Encode each player on a team from their PBP plays.

        Args:
            player_play_embs:  (B, P, T, play_dim)
            player_play_masks: (B, P, T) bool — True = padding
        Returns:
            (B, P, d_player)
        """
        B, P, T, _ = player_play_embs.shape
        flat_plays = player_play_embs.reshape(B * P, T, -1)
        flat_masks = player_play_masks.reshape(B * P, T)
        player_embs = self.pbp_pooler(flat_plays, flat_masks)
        return player_embs.reshape(B, P, -1)

    def forward(
        self,
        plays_a, play_mask_a, roster_mask_a,
        plays_b, play_mask_b, roster_mask_b,
    ):
        """Full forward pass — player-level all the way through.

        Args:
            plays_a:       (B, Pa, T, play_dim) — Team A per-player play embeddings
            play_mask_a:   (B, Pa, T) — True = padding play
            roster_mask_a: (B, Pa) — True = player unavailable
            plays_b:       (B, Pb, T, play_dim)
            play_mask_b:   (B, Pb, T)
            roster_mask_b: (B, Pb)
        Returns:
            logit:  (B,)
            margin: (B,)
        """
        # 1. Encode each player from their plays
        a_embs = self.encode_players(plays_a, play_mask_a)  # (B, Pa, d)
        b_embs = self.encode_players(plays_b, play_mask_b)  # (B, Pb, d)

        # 2. Team self-attention: teammates attend to each other
        for layer in self.team_self_layers:
            a_embs = layer(a_embs, key_padding_mask=roster_mask_a)
            b_embs = layer(b_embs, key_padding_mask=roster_mask_b)
        a_embs = self.self_norm(a_embs)
        b_embs = self.self_norm(b_embs)

        # 3. Cross-attention: A's players attend to B's players and vice versa
        a_cross = a_embs
        b_cross = b_embs
        for layer in self.cross_layers:
            a_cross = layer(a_cross, kv=b_cross, key_padding_mask=roster_mask_b)
            b_cross = layer(b_cross, kv=a_cross, key_padding_mask=roster_mask_a)
        a_cross = self.cross_norm(a_cross)
        b_cross = self.cross_norm(b_cross)

        # 4. Add side embeddings and concatenate all players
        a_with_side = a_cross + self.side_emb.weight[0]  # (B, Pa, d)
        b_with_side = b_cross + self.side_emb.weight[1]  # (B, Pb, d)
        all_players = torch.cat([a_with_side, b_with_side], dim=1)  # (B, Pa+Pb, d)
        all_mask = torch.cat([roster_mask_a, roster_mask_b], dim=1)  # (B, Pa+Pb)

        # 5. Prediction attention: learned query attends over all players
        B = all_players.shape[0]
        query = self.pred_query.expand(B, -1, -1)

        # Safety: if all players are masked for a batch item, unmask
        # the first position to avoid NaN from attention over empty keys
        safe_mask = all_mask.clone()
        all_empty = safe_mask.all(dim=1)
        if all_empty.any():
            safe_mask[all_empty, 0] = False

        pred_ctx, _ = self.pred_attn(
            query, all_players, all_players, key_padding_mask=safe_mask,
        )
        pred_ctx = self.pred_norm(pred_ctx.squeeze(1))  # (B, d)

        logit = self.head(pred_ctx).squeeze(-1)
        margin = self.margin_head(pred_ctx).squeeze(-1)
        return logit, margin

    def matchup_embedding(
        self,
        plays_a, play_mask_a, roster_mask_a,
        plays_b, play_mask_b, roster_mask_b,
    ):
        """Extract penultimate embedding for AutoGluon features."""
        # Run full forward to get pred_ctx, then extract penultimate layer
        a_embs = self.encode_players(plays_a, play_mask_a)
        b_embs = self.encode_players(plays_b, play_mask_b)

        for layer in self.team_self_layers:
            a_embs = layer(a_embs, key_padding_mask=roster_mask_a)
            b_embs = layer(b_embs, key_padding_mask=roster_mask_b)
        a_embs = self.self_norm(a_embs)
        b_embs = self.self_norm(b_embs)

        a_cross = a_embs
        b_cross = b_embs
        for layer in self.cross_layers:
            a_cross = layer(a_cross, kv=b_cross, key_padding_mask=roster_mask_b)
            b_cross = layer(b_cross, kv=a_cross, key_padding_mask=roster_mask_a)
        a_cross = self.cross_norm(a_cross)
        b_cross = self.cross_norm(b_cross)

        a_with_side = a_cross + self.side_emb.weight[0]
        b_with_side = b_cross + self.side_emb.weight[1]
        all_players = torch.cat([a_with_side, b_with_side], dim=1)
        all_mask = torch.cat([roster_mask_a, roster_mask_b], dim=1)

        B = all_players.shape[0]
        query = self.pred_query.expand(B, -1, -1)
        pred_ctx, _ = self.pred_attn(
            query, all_players, all_players, key_padding_mask=all_mask,
        )
        pred_ctx = self.pred_norm(pred_ctx.squeeze(1))

        # Penultimate layer of head
        x = self.head[0](pred_ctx)  # Linear
        x = self.head[1](x)         # GELU
        return x
