import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from crystalformer.src.wyckoff import wmax_table, dof0_table
from crystalformer.src.attention import MultiHeadAttentionTorch


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)


class CrystalFormerTransformer(nn.Module):
    def __init__(
        self,
        Nf,
        Kx,
        Kl,
        n_max,
        h0_size,
        num_layers,
        num_heads,
        key_size,
        model_size,
        embed_size,
        atom_types,
        wyck_types,
        dropout_rate,
        widening_factor=4,
        sigmamin=1e-3,
    ):
        super().__init__()

        self.Nf = Nf
        self.Kx = Kx
        self.Kl = Kl
        self.n_max = n_max
        self.atom_types = atom_types
        self.wyck_types = wyck_types
        self.coord_types = 3 * Kx
        self.lattice_types = Kl + 2 * 6 * Kl
        self.output_size = max(
            atom_types + self.lattice_types,
            self.coord_types,
            wyck_types,
        )
        self.sigmamin = sigmamin
        self.dropout_rate = dropout_rate

        # Embeddings
        self.g_embeddings = nn.Embedding(230, embed_size)
        self.w_embeddings = nn.Embedding(wyck_types, embed_size)
        self.a_embeddings = nn.Embedding(atom_types, embed_size)

        self.positional_embeddings = nn.Parameter(
            torch.randn(5 * n_max, model_size) * 0.01
        )

        # Input projections
        self.hW_proj = nn.Linear(embed_size + embed_size + 1, model_size)
        self.hA_proj = nn.Linear(embed_size + embed_size, model_size)

        self.coord_proj = nn.Linear(embed_size + 2 * Nf, model_size)

        # Transformer blocks
        self.attn_blocks = nn.ModuleList()
        self.ff_blocks = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.ln2 = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_blocks.append(
                MultiHeadAttentionTorch(
                    num_heads=num_heads,
                    key_size=key_size,
                    model_size=model_size,
                    dropout_rate=dropout_rate,
                )
            )
            self.ff_blocks.append(
                nn.Sequential(
                    nn.Linear(model_size, widening_factor * model_size),
                    nn.GELU(),
                    nn.Linear(widening_factor * model_size, model_size),
                )
            )
            self.ln1.append(LayerNorm(model_size))
            self.ln2.append(LayerNorm(model_size))

        self.final_ln = LayerNorm(model_size)
        self.out_proj = nn.Linear(model_size, self.output_size)

    def renormalize(self, h_x):
        n = h_x.shape[0]
        x_logit, x_loc, x_kappa = torch.split(
            h_x[:, : self.coord_types], self.Kx, dim=-1
        )
        x_logit = x_logit - torch.logsumexp(x_logit, dim=1, keepdim=True)
        x_kappa = F.softplus(x_kappa)
        pad = torch.zeros(n, self.output_size - self.coord_types, device=h_x.device)
        return torch.cat([x_logit, x_loc, x_kappa, pad], dim=-1)

    def forward(self, G, XYZ, A, W, M, is_train=True):
        """
        G: (scalar) space group id (1–230)
        XYZ: (n, 3)
        A: (n,)
        W: (n,)
        M: (n,)
        """

        device = XYZ.device
        n = XYZ.shape[0]

        g_emb = self.g_embeddings(G - 1)
        w_emb = self.w_embeddings(W)
        a_emb = self.a_embeddings(A)

        # First atom logits
        w_max = wmax_table[G - 1]
        w_logit = torch.zeros(self.wyck_types, device=device)
        w_mask = (torch.arange(self.wyck_types, device=device) > 0) & (
            torch.arange(self.wyck_types, device=device) <= w_max
        )
        w_logit[~w_mask] = -1e10
        w_logit = w_logit - torch.logsumexp(w_logit, dim=0)

        h0 = torch.cat(
            [w_logit.unsqueeze(0), torch.zeros(1, self.output_size - self.wyck_types, device=device)],
            dim=-1,
        )

        if n == 0:
            return h0

        # Build token embeddings
        hW = torch.cat(
            [
                g_emb.unsqueeze(0).repeat(n, 1),
                w_emb,
                M.float().unsqueeze(1),
            ],
            dim=1,
        )
        hW = self.hW_proj(hW)

        hA = torch.cat(
            [
                g_emb.unsqueeze(0).repeat(n, 1),
                a_emb,
            ],
            dim=1,
        )
        hA = self.hA_proj(hA)

        coords = []
        for coord in XYZ.T:
            feats = [g_emb.unsqueeze(0).repeat(n, 1)]
            for f in range(1, self.Nf + 1):
                feats.append(torch.sin(2 * math.pi * coord.unsqueeze(1) * f))
                feats.append(torch.cos(2 * math.pi * coord.unsqueeze(1) * f))
            coords.append(self.coord_proj(torch.cat(feats, dim=1)))

        h = torch.stack([hW, hA, coords[0], coords[1], coords[2]], dim=1)
        h = h.view(5 * n, -1)

        h = h + self.positional_embeddings[: 5 * n]

        mask = torch.tril(torch.ones(5 * n, 5 * n, device=device)).unsqueeze(0)

        for attn, ff, ln1, ln2 in zip(
            self.attn_blocks, self.ff_blocks, self.ln1, self.ln2
        ):
            h = h + attn(ln1(h), ln1(h), ln1(h), mask=mask)
            h = h + ff(ln2(h))

        h = self.final_ln(h)
        h = self.out_proj(h)

        h = h.view(n, 5, -1)
        h_al, h_x, h_y, h_z, w_logit = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4]

        h_x = self.renormalize(h_x)
        h_y = self.renormalize(h_y)
        h_z = self.renormalize(h_z)

        return torch.cat([h0, h_al, h_x, h_y, h_z], dim=0)
