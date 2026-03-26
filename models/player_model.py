"""Player-level neural network for matchup-aware game prediction.

Architecture
============
Each team's rotation (~8 players) is individually encoded, then:

  1. Player Encoder (shared MLP)
     Per-player stats (BPM, ORtg, usage, height, ...) → d-dim embedding.
     Shared weights across all players and teams.

  2. Team Self-Attention (N layers)
     Players attend to teammates → captures chemistry/positional balance.
     A rim-protector's value depends on the perimeter defenders beside him.

  3. Cross-Team Attention (N layers)
     Team A's players attend to Team B's players → captures matchup dynamics.
     A slow center's value changes when facing a small-ball lineup.

  4. Prediction Head
     Concatenate team embeddings + matchup embedding → P(Team A wins).

Outputs for the tabular ensemble
=================================
  - Team embeddings (team-level features, pipeline creates _A/_B/_delta)
  - Matchup prediction (matchup-level feature)
"""

import torch
import torch.nn as nn


PLAYER_FEATURES = [
    "min_pct", "ortg", "usg", "efg", "ts_pct",
    "orb_pct", "drb_pct", "ast_pct", "to_pct",
    "bpm", "height_inches", "class_num",
]

N_PLAYER_FEATURES = len(PLAYER_FEATURES)
MAX_PLAYERS = 8


class AttentionBlock(nn.Module):
    """Pre-norm transformer block: attention + FFN with residual connections."""

    def __init__(self, embed_dim, n_heads, dropout=0.1, is_cross=False,
                 ffn_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.is_cross = is_cross
        if is_cross:
            self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed-forward network (standard transformer FFN)
        ffn_dim = embed_dim * ffn_mult
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, kv=None, key_padding_mask=None):
        # Attention with residual
        residual = x
        x = self.norm(x)
        if self.is_cross:
            kv = self.norm_kv(kv)
            out, _ = self.attn(x, kv, kv, key_padding_mask=key_padding_mask)
        else:
            out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + out

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PlayerMatchupModel(nn.Module):
    def __init__(
        self,
        player_feat_dim: int = N_PLAYER_FEATURES,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Shared player encoder
        hidden = embed_dim * 2
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Within-team self-attention layers
        self.team_self_layers = nn.ModuleList([
            AttentionBlock(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Cross-team attention layers
        self.cross_layers = nn.ModuleList([
            AttentionBlock(embed_dim, n_heads, dropout, is_cross=True)
            for _ in range(n_layers)
        ])

        # Final norms
        self.team_final_norm = nn.LayerNorm(embed_dim)
        self.cross_final_norm = nn.LayerNorm(embed_dim)

        # Prediction head: team_a_emb + team_b_emb + matchup_emb
        head_in = embed_dim * 3
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

        # Auxiliary: score margin prediction (provides richer gradient signal)
        self.margin_head = nn.Linear(head_in, 1)

    def _pool(self, x, mask):
        """Mean-pool over non-masked positions."""
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)

    def encode_team(self, players, mask=None):
        """Encode a team's players → (team_embedding, player_embeddings)."""
        emb = self.player_encoder(players)
        for layer in self.team_self_layers:
            emb = layer(emb, key_padding_mask=mask)
        emb = self.team_final_norm(emb)
        team_emb = self._pool(emb, mask)
        return team_emb, emb

    def forward(self, team_a, team_b, mask_a=None, mask_b=None):
        """Full forward pass.

        Returns:
            logit: (batch,) raw logit for P(Team A wins)
            margin: (batch,) predicted score margin (Team A - Team B)
            team_a_emb: (batch, embed_dim)
            team_b_emb: (batch, embed_dim)
            matchup_emb: (batch, embed_dim)
        """
        team_a_emb, a_player_embs = self.encode_team(team_a, mask_a)
        team_b_emb, b_player_embs = self.encode_team(team_b, mask_b)

        # Cross-attention: each A player attends to all B players
        cross_out = a_player_embs
        for layer in self.cross_layers:
            cross_out = layer(cross_out, kv=b_player_embs, key_padding_mask=mask_b)
        cross_out = self.cross_final_norm(cross_out)
        matchup_emb = self._pool(cross_out, mask_a)

        combined = torch.cat([team_a_emb, team_b_emb, matchup_emb], dim=-1)
        logit = self.head(combined).squeeze(-1)
        margin = self.margin_head(combined).squeeze(-1)

        return logit, margin, team_a_emb, team_b_emb, matchup_emb
