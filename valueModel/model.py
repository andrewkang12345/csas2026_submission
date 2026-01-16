import torch
from torch import nn


class ValueTransformer(nn.Module):
    """
    Transformer-based state value network for curling.

    Inputs:
      x: (B, 24) normalized stone coords for 12 stones
      c: (B, cond_dim) context features, now:
         [shot_norm, is_hammer, team_order]

    Tokens:
      - one global token from c
      - 12 stone tokens from (x,y) with stone-index embedding

    Output:
      - scalar value per row/team perspective
    """

    def __init__(
        self,
        input_dim,    # 24
        cond_dim,     # 3
        hidden_dim=256,
        num_stones=12,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_stones = num_stones
        self.pos_dim = num_stones * 2

        if input_dim != self.pos_dim:
            raise ValueError(
                f"Expected input_dim={self.pos_dim} (12 stones * 2 coords), got {input_dim}"
            )

        d_model = hidden_dim
        self.d_model = d_model

        self.stone_proj = nn.Linear(2, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        self.stone_index_embed = nn.Embedding(num_stones, d_model)
        self.register_buffer(
            "stone_indices",
            torch.arange(num_stones, dtype=torch.long),
            persistent=False,
        )

        self.global_token_param = nn.Parameter(torch.randn(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,     # IMPORTANT: avoids perf warning, simpler shapes
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, c):
        """
        x: (B, 24)
        c: (B, cond_dim)

        returns: (B, 1)
        """
        B = x.size(0)
        device = x.device

        stones = x.view(B, self.num_stones, 2)            # (B, 12, 2)
        stone_tokens = self.stone_proj(stones)            # (B, 12, d)

        stone_idx = self.stone_indices.to(device)         # (12,)
        stone_idx_emb = self.stone_index_embed(stone_idx) # (12, d)
        stone_idx_emb = stone_idx_emb.unsqueeze(0).expand(B, -1, -1)
        stone_tokens = stone_tokens + stone_idx_emb

        cond_token = self.cond_proj(c).unsqueeze(1)       # (B, 1, d)
        global_token = cond_token + self.global_token_param.expand(B, -1, -1)

        tokens = torch.cat([global_token, stone_tokens], dim=1)  # (B, 13, d)
        enc_out = self.encoder(tokens)                            # (B, 13, d)

        global_out = enc_out[:, 0, :]                             # (B, d)
        return self.value_head(global_out)                        # (B, 1)
