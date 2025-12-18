import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from crystalformer.src.wyckoff import wmax_table, dof0_table

class HaikuMultiHeadAttention(nn.Module):
    """
    Re-implementation of Haiku's MultiHeadAttention in PyTorch.
    Allows arbitrary key_size which decouples head_dim from model_size.
    """
    def __init__(self, num_heads, key_size, model_size, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        
        # Projections
        self.q_proj = nn.Linear(model_size, num_heads * key_size, bias=True)
        self.k_proj = nn.Linear(model_size, num_heads * key_size, bias=True)
        self.v_proj = nn.Linear(model_size, num_heads * key_size, bias=True)
        
        # Output project
        self.o_proj = nn.Linear(num_heads * key_size, model_size, bias=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = key_size ** -0.5

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape: (B, L, H, K) -> (B, H, L, K)
        q = q.view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        output = torch.matmul(probs, v)
        
        # Reshape back: (B, H, L, K) -> (B, L, H*K)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, key_size, model_size, dropout_rate=0.1, widening_factor=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_size)
        self.ln2 = nn.LayerNorm(model_size)
        self.attn = HaikuMultiHeadAttention(num_heads, key_size, model_size, dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(model_size, widening_factor * model_size),
            nn.GELU(),
            nn.Linear(widening_factor * model_size, model_size)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, is_train=True):
        h_norm = self.ln1(x)
        attn_out = self.attn(h_norm, h_norm, h_norm, mask=mask)
        if is_train: attn_out = self.dropout(attn_out)
        x = x + attn_out

        h_norm = self.ln2(x)
        mlp_out = self.mlp(h_norm)
        if is_train: mlp_out = self.dropout(mlp_out)
        x = x + mlp_out
        return x

class CrystalTransformer(nn.Module):
    def __init__(self, Nf, Kx, Kl, n_max, h0_size, num_layers, num_heads, key_size, 
                 model_size, embed_size, atom_types, wyck_types, dropout_rate, 
                 attn_dropout=0.1, widening_factor=4, sigmamin=1e-3):
        super().__init__()
        
        self.Nf = Nf
        self.Kx = Kx
        self.Kl = Kl
        self.n_max = n_max
        self.h0_size = h0_size
        self.model_size = model_size
        self.embed_size = embed_size
        self.atom_types = atom_types
        self.wyck_types = wyck_types
        self.sigmamin = sigmamin
        
        self.coord_types = 3 * Kx
        self.lattice_types = Kl + 2 * 6 * Kl
        self.output_size = max(atom_types + self.lattice_types, self.coord_types, wyck_types)

        self.g_embeddings = nn.Embedding(230, embed_size)
        self.w_embeddings = nn.Embedding(wyck_types, embed_size)
        self.a_embeddings = nn.Embedding(atom_types, embed_size)
        
        if h0_size > 0:
            self.h0_mlp = nn.Sequential(
                nn.Linear(embed_size, h0_size),
                nn.GELU(),
                nn.Linear(h0_size, wyck_types)
            )
        else:
            self.w_params = nn.Parameter(torch.randn(230, wyck_types) * 0.01)

        self.fc_hW = nn.Linear(2 * embed_size + 1, model_size)
        self.fc_hA = nn.Linear(2 * embed_size, model_size)
        
        # --- UPDATE: Split Coordinate Projections (Matches JAX Linear_4, 5, 6) ---
        self.fc_hX = nn.Linear(embed_size + 2 * Nf, model_size)
        self.fc_hY = nn.Linear(embed_size + 2 * Nf, model_size)
        self.fc_hZ = nn.Linear(embed_size + 2 * Nf, model_size)

        self.layers = nn.ModuleList([
            TransformerBlock(num_heads, key_size, model_size, attn_dropout, widening_factor)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_size)
        self.output_proj = nn.Linear(model_size, self.output_size)

    def renormalize_coord(self, h_x):
        relevant = h_x[..., :self.coord_types]
        x_logit, x_loc, x_kappa = torch.split(relevant, [self.Kx, self.Kx, self.Kx], dim=-1)
        x_logit = torch.log_softmax(x_logit, dim=-1)
        x_kappa = F.softplus(x_kappa)
        padding = torch.zeros(h_x.shape[:-1] + (self.output_size - self.coord_types,), device=h_x.device)
        return torch.cat([x_logit, x_loc, x_kappa, padding], dim=-1)

    def forward(self, G, XYZ, A, W, M, is_train=True):
        device = self.g_embeddings.weight.device
        
        if not isinstance(G, torch.Tensor): G = torch.tensor(G, device=device)
        XYZ = XYZ.to(device)
        A = A.to(device)
        W = W.to(device)
        M = M.to(device)

        is_batched = XYZ.ndim == 3
        if not is_batched:
            G = G.reshape(1)
            XYZ = XYZ.unsqueeze(0)
            A = A.unsqueeze(0)
            W = W.unsqueeze(0)
            M = M.unsqueeze(0)

        batch_size = XYZ.shape[0]
        n_atoms = XYZ.shape[1]
        
        G_idx = G.long() - 1
        g_emb = self.g_embeddings(G_idx) 
        
        wmax_tensor = torch.tensor(wmax_table, device=device) 
        w_max_val = wmax_tensor[G_idx] 

        if self.h0_size > 0:
            w_logit_0 = self.h0_mlp(g_emb) 
        else:
            w_logit_0 = self.w_params[G_idx] 

        w_range = torch.arange(self.wyck_types, device=device).unsqueeze(0) 
        w_mask_0 = (w_range > 0) & (w_range <= w_max_val.unsqueeze(1))
        
        w_logit_0 = torch.where(w_mask_0, w_logit_0, w_logit_0 - 1e10)
        w_logit_0 = torch.log_softmax(w_logit_0, dim=1)
        
        h0 = torch.cat([
            w_logit_0.unsqueeze(1), 
            torch.zeros((batch_size, 1, self.output_size - self.wyck_types), device=device)
        ], dim=-1) 
        
        if n_atoms == 0:
            return h0 if is_batched else h0.squeeze(0)

        g_emb_exp = g_emb.unsqueeze(1).expand(-1, n_atoms, -1)
        
        w_emb = self.w_embeddings(W.long()) 
        hW_in = torch.cat([g_emb_exp, w_emb, M.float().unsqueeze(-1)], dim=-1)
        hW = self.fc_hW(hW_in) 

        a_emb = self.a_embeddings(A.long())
        hA_in = torch.cat([g_emb_exp, a_emb], dim=-1)
        hA = self.fc_hA(hA_in)

        X, Y, Z = XYZ[:,:,0], XYZ[:,:,1], XYZ[:,:,2] 
        
        def fourier_encode(coords, Nf):
            freqs = torch.arange(1, Nf+1, device=device).float()
            args = 2 * math.pi * coords.unsqueeze(-1) * freqs.view(1, 1, -1)
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        hX_f = fourier_encode(X, self.Nf)
        hY_f = fourier_encode(Y, self.Nf)
        hZ_f = fourier_encode(Z, self.Nf)
        
        # --- UPDATE: Use Split Coordinate Projections ---
        hX = self.fc_hX(torch.cat([g_emb_exp, hX_f], dim=-1))
        hY = self.fc_hY(torch.cat([g_emb_exp, hY_f], dim=-1))
        hZ = self.fc_hZ(torch.cat([g_emb_exp, hZ_f], dim=-1))
        
        h = torch.stack([hW, hA, hX, hY, hZ], dim=2)
        h = h.reshape(batch_size, 5 * n_atoms, self.model_size)
        
        seq_len = 5 * n_atoms
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        float_mask = torch.zeros(seq_len, seq_len, device=device)
        float_mask = float_mask.masked_fill(mask, float('-inf'))
        
        for layer in self.layers:
            h = layer(h, mask=float_mask, is_train=is_train)
            
        h = self.final_norm(h)
        h = self.output_proj(h) 
        
        h = h.reshape(batch_size, n_atoms, 5, -1)
        
        h_al = h[:, :, 0, :]
        h_x = h[:, :, 1, :]
        h_y = h[:, :, 2, :]
        h_z = h[:, :, 3, :]
        w_logit_seq = h[:, :, 4, :]
        
        h_x = self.renormalize_coord(h_x)
        h_y = self.renormalize_coord(h_y)
        h_z = self.renormalize_coord(h_z)
        
        a_logit = h_al[..., :self.atom_types]
        w_logit = w_logit_seq[..., :self.wyck_types]
        
        # --- Masking Logic ---
        w_range = torch.arange(1, self.wyck_types, device=device).view(1, 1, -1)
        W_uns = W.unsqueeze(-1) 
        
        mask_less_equal = w_range < W_uns
        mask_less = w_range <= W_uns
        
        dof0_gpu = torch.tensor(dof0_table, device=device) 
        G_exp = G_idx.unsqueeze(1).expand(-1, n_atoms)
        is_dof0 = dof0_gpu[G_exp, W.long()] 
        
        w_mask_1 = torch.where(is_dof0.unsqueeze(-1), mask_less, mask_less_equal)
        pad_col = torch.zeros(batch_size, n_atoms, 1, device=device, dtype=torch.bool)
        w_mask_1 = torch.cat([pad_col, w_mask_1], dim=-1)
        
        w_logit = w_logit - torch.where(w_mask_1, torch.tensor(1e10, device=device), torch.tensor(0.0, device=device))
        w_logit = torch.log_softmax(w_logit, dim=-1)
        
        idx_0_mask = torch.zeros_like(w_logit, dtype=torch.bool)
        idx_0_mask[..., 0] = True
        w_is_0_mask = (W == 0).unsqueeze(-1).expand_as(w_logit)
        
        w_logit = torch.where(idx_0_mask & w_is_0_mask, w_logit + 1e10, w_logit)
        w_logit = torch.log_softmax(w_logit, dim=-1)
        
        w_indices = torch.arange(self.wyck_types, device=device).view(1, 1, -1)
        w_mask_3 = w_indices <= w_max_val.view(-1, 1, 1)
        w_logit = torch.where(w_mask_3, w_logit, w_logit - 1e10)
        w_logit = torch.log_softmax(w_logit, dim=-1)
        
        w_gt_0 = (W > 0).unsqueeze(-1)
        w_eq_0 = (W == 0).unsqueeze(-1)
        
        a_mask = torch.cat([
            w_gt_0,
            w_eq_0.repeat(1, 1, self.atom_types - 1)
        ], dim=-1)
        
        a_logit = a_logit - torch.where(a_mask, torch.tensor(1e10, device=device), torch.tensor(0.0, device=device))
        a_logit = torch.log_softmax(a_logit, dim=-1)
        
        w_logit_final = torch.cat([
            w_logit,
            torch.zeros(batch_size, n_atoms, self.output_size - self.wyck_types, device=device)
        ], dim=-1)
        
        l_part = h_al[..., self.atom_types : self.atom_types + self.lattice_types]
        l_logit, mu, sigma = torch.split(l_part, [self.Kl, 6*self.Kl, 6*self.Kl], dim=-1)
        
        l_logit = torch.log_softmax(l_logit, dim=-1)
        sigma = F.softplus(sigma) + self.sigmamin
        
        h_al_final = torch.cat([
            a_logit,
            l_logit,
            mu,
            sigma,
            torch.zeros(batch_size, n_atoms, self.output_size - self.atom_types - self.lattice_types, device=device)
        ], dim=-1)
        
        h_final = torch.stack([
            h_al_final,
            h_x,
            h_y,
            h_z,
            w_logit_final
        ], dim=2) 
        
        h_final = h_final.reshape(batch_size, 5*n_atoms, self.output_size)
        
        result = torch.cat([h0, h_final], dim=1) 
        
        if not is_batched:
            return result.squeeze(0)
            
        return result

def make_transformer(key, Nf, Kx, Kl, n_max, h0_size, num_layers, num_heads, key_size, 
                     model_size, embed_size, atom_types, wyck_types, dropout_rate, 
                     attn_dropout=0.1, widening_factor=4, sigmamin=1e-3):
    return CrystalTransformer(
        Nf=Nf, Kx=Kx, Kl=Kl, n_max=n_max, h0_size=h0_size, 
        num_layers=num_layers, num_heads=num_heads, key_size=key_size, 
        model_size=model_size, embed_size=embed_size, 
        atom_types=atom_types, wyck_types=wyck_types, 
        dropout_rate=dropout_rate, attn_dropout=attn_dropout, 
        widening_factor=widening_factor, sigmamin=sigmamin
    )
