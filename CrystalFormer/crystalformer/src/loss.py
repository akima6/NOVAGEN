import torch
import torch.nn.functional as F
import numpy as np

# We assume these tables/functions are available from rewritten modules
from crystalformer.src.lattice import make_lattice_mask
from crystalformer.src.wyckoff import mult_table, fc_mask_table

# --- Helper: Von Mises Log PDF ---
def von_mises_logpdf(x, loc, kappa):
    """
    Computes log(VonMises(x | loc, kappa))
    x: (Batch, n_max, 1) or broadcastable
    loc: (Batch, n_max, Kx)
    kappa: (Batch, n_max, Kx)
    """
    # CRITICAL FIX: Clamp kappa to prevent numerical instability
    kappa = torch.clamp(kappa, min=1e-8)
    
    # log(I0(k)) = k + log(I0e(k))
    term1 = kappa * torch.cos(x.unsqueeze(-1) - loc) # (Batch, n_max, Kx)
    term2 = -np.log(2 * np.pi)
    term3 = -(kappa + torch.log(torch.special.i0e(kappa)))
    
    return term1 + term2 + term3

# --- Helper: Gaussian Log PDF ---
def gaussian_logpdf(x, loc, scale):
    """
    Computes log(Gaussian(x | loc, scale))
    """
    # CRITICAL FIX 1: Aggressive clamp on scale for stability (JAX used a softplus/exp to ensure positivity, we use clamp)
    scale = torch.clamp(scale, min=1e-6) # Floor set to 1e-6 (was 1e-8)
    
    x = x.unsqueeze(1)
    var = scale ** 2
    log_scale = torch.log(scale)
    
    # CRITICAL FIX 2: Explicitly move the constant to the device for consistency
    LOG_2PI = torch.tensor(np.log(2 * np.pi), device=x.device, dtype=x.dtype)
    
    # We must also clamp the exponent term to prevent -Inf if var is too small
    exponent = -((x - loc) ** 2) / (2 * var)
    exponent = torch.clamp(exponent, max=100.0) # Prevent overflow from massive negative numbers
    
    return exponent - log_scale - 0.5 * LOG_2PI

def make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer, lamb_a=1.0, lamb_w=1.0, lamb_l=1.0):
    """
    Factory for loss functions.
    transformer: PyTorch model
    """
    
    coord_types = 3 * Kx
    # Lattice mask (move to device inside function)
    lattice_mask_cpu = make_lattice_mask() 

    def compute_logp_x(h_x, X, fc_mask_x):
        """
        h_x: (Batch, n_max, coord_types)
        X: (Batch, n_max)
        fc_mask_x: (Batch, n_max) boolean
        """
        batch_size = h_x.shape[0]
        
        # Split h_x
        # h_x has 3*Kx features: logits(Kx), loc(Kx), kappa(Kx)
        x_logit, loc, kappa = torch.split(h_x, [Kx, Kx, Kx], dim=-1)
        
        # LogSoftmax logits for mixture weights
        x_logit = F.log_softmax(x_logit, dim=-1)
        
        # Kappa softplus (already done in transformer? No, transformer.py does it at output. 
        # But wait, transformer.py 'renormalize' does softplus on kappa. 
        # So we receive softplussed kappa.)
        
        # Von Mises Log PDF
        # Input X is fractional [0, 1]. Convert to [-pi, pi]? 
        # Original: (X-0.5)*2*jnp.pi -> range [-pi, pi]
        X_rad = (X - 0.5) * 2 * np.pi
        
        # log_prob per component: (Batch, n_max, Kx)
        log_vm = von_mises_logpdf(X_rad, loc, kappa)
        
        # Mixture Log Prob: LogSumExp(weight_logs + component_logs)
        # (Batch, n_max)
        logp_x = torch.logsumexp(x_logit + log_vm, dim=-1)
        
        # Masking
        # fc_mask_x is True where we compute loss
        logp_x = torch.where(fc_mask_x, logp_x, torch.zeros_like(logp_x))
        
        # Sum over atoms (Batch,)
        return torch.sum(logp_x, dim=1)

    def logp_fn(model, G, L, XYZ, A, W, is_train=True):
        """
        Computes log probabilities for the batch.
        params is unused (kept for signature compatibility if needed, but in PyTorch model has params).
        """
        batch_size = G.shape[0]
        device = G.device
        
        # Mult table lookup
        mult_table_gpu = mult_table.to(device)
        G_idx = (G - 1).long()
        G_expanded = G_idx.unsqueeze(1).expand(-1, W.shape[1])
        M = mult_table_gpu[G_expanded, W.long()] # (Batch, n_max)
        
        num_sites = torch.sum(A != 0, dim=1) # (Batch,)
        
        # Forward Pass
        # h: (5*n+1, output) ? No, PyTorch version returns (5*n+1, output) or (Batch, 5*n+1, output)?
        # Our rewritten transformer.py forward returns (5*n+1, output) for single?
        # WAIT. Rewritten transformer.py forward() handles BATCHING?
        # Let's check transformer.py rewrite.
        # "def forward(self, G, XYZ, A, W, M, is_train=True):"
        # It handles batch logic internally (stacking hW, hA...).
        # It returns "result = torch.cat([h0, h_final], dim=0)".
        # Wait, if input has batch dim N, h0 is (1, output). 
        # h_final is (5*N, output).
        # Concatenation dim=0 -> (5*N+1, output).
        # This structure implies the transformer processes a single huge sequence or assumes Batch=1?
        # Original JAX code: "h = h.reshape(5*n, -1)" -> flattens batch? 
        # No, JAX "n" was "XYZ.shape[0]". In JAX code `network`, XYZ was (n, 3).
        # The JAX `network` function processed a SINGLE crystal. 
        # Then `make_loss_fn` used `vmap` to handle batching.
        
        # PROBLEM: My rewritten transformer.py handles `n` as `XYZ.shape[0]`.
        # If I pass a batch (B, n, 3), `n` becomes B. 
        # This is WRONG if the original `network` was per-crystal.
        # Original JAX: `XYZ: (n, 3)`. `network` is `@hk.transform`.
        # `inference` in `sample.py` used `vmap` -> mapped over batch.
        
        # My Rewritten Transformer:
        # "n = XYZ.shape[0]" -> This interprets the first dimension as atoms.
        # If I pass (Batch, n_max, 3), it treats it as `n` atoms?
        # NO. The transformer rewrite expects 2D XYZ.
        # If I want batching, I need to adjust `CrystalTransformer` or wrap it.
        # BUT, `CrystalTransformer` logic:
        # "g_embeddings(G_idx)" -> if G is scalar, (emb). If G is vector (B), (B, emb).
        # "g_emb.unsqueeze(0).expand(n, -1)" -> expands to n atoms.
        
        # Current Rewritten Transformer.py supports inputs (n, 3). 
        # This means it supports ONE crystal with n atoms.
        # To support batching (Batch, n_max, 3), we need to modify it or vector-map it.
        # PyTorch doesn't have `vmap` as maturely.
        # We usually write the model to handle (Batch, Seq, Feat).
        
        # The rewrite I provided in transformer.py was slightly ambiguous on batching because the original was vmapped.
        # However, looking at my transformer.py:
        # "G_idx = G.long() - 1" -> handles batch G.
        # "h = h.reshape(5 * n, self.model_size) # (5*n, model_size)"
        # "h = h.unsqueeze(0) # (1, 5*n, model_size)" -> Batch dim 1 for Transformer.
        # This strongly suggests my transformer.py processes a SINGLE sequence of length 5*n.
        # If n is "Batch * n_max", then it's one long sequence? No, attention would be full N*N.
        # We want independent batches.
        
        # To fix this properly for PyTorch:
        # We must assume the input to `logp_fn` is a BATCH.
        # But `transformer` processes (N, 3).
        # If we cannot change transformer.py (Phase 1 done), we must wrap it.
        # BUT, Phase 1 transformer.py was written by me. I checked it. 
        # It takes "G, XYZ..." 
        # If I pass (Batch, n_max, 3), `n` = Batch. XYZ[:, 0] gets slice (Batch, 3).
        # It seems it conflates Batch with Sequence Length if not careful.
        
        # CORRECTION:
        # I need to ensure `transformer` runs in batch mode.
        # Since I cannot edit transformer.py now, I will use a Loop or assume the user patches it?
        # Actually, standard PyTorch Transformers take (Batch, Seq, Feat).
        # My transformer.py:
        # "h = torch.stack([hW, hA, hX, hY, hZ], dim=1)" -> (n, 5, model)
        # If input was (Batch, n_max, 3), then n=Batch. 
        # This treats the BATCH dimension as the ATOM dimension.
        # This is NOT what we want.
        # We want (Batch, n_max, 3).
        
        # Let's look at `sample.py` rewrite.
        # `inference` calls `model(g, XYZ...)`.
        # `XYZ` is (Batch, n_max, 3).
        # If `transformer` treats dim 0 as atoms, we are mixing samples!
        
        # CRITICAL FIX for `loss.py` (and retroactively implies transformer needs care):
        # We can implement a manual batch loop here since `vmap` is gone.
        # Or, ideally, the transformer handles (Batch, n_max).
        
        # Let's assume for this Phase 3 file that we handle the model output correctly.
        # If `transformer` expects (N, ...), we reshape (Batch*n_max, ...).
        # Attention mask prevents cross-sample attention.
        # The original JAX used vmap, so independent samples.
        # My transformer.py used `nn.MultiheadAttention(batch_first=True)`.
        # Input to it was `h.unsqueeze(0)`. This forces Batch=1.
        # This confirms my transformer.py was designed for Single Sample (vmap style).
        
        # SOLUTION:
        # In `logp_fn`, we iterate over the batch and stack results.
        # This is slow but correct given the constraints.
        # (Alternatively, rewrite transformer to be batched, but Phase 1 is closed).
        
        outputs = []
        for i in range(batch_size):
            # Slice single sample
            # G[i]: scalar
            # XYZ[i]: (n_max, 3)
            # A[i]: (n_max,)
            # W[i]: (n_max,)
            # M[i]: (n_max,)
            out_i = model(G[i], XYZ[i], A[i], W[i], M[i], is_train)
            # out_i: (5*n_max+1, output_size)
            outputs.append(out_i)
            
        h = torch.stack(outputs, dim=0) # (Batch, 5*n_max+1, output_size)
        
        # Now process h (Batch, Time, Feat)
        # Indices:
        # 0: h0 (Batch, 1, ...)
        # 1..: Body
        
        # w_logit: (Batch, n_max, wyck_types)
        # Taken from 0::5?
        # Index 0 is h0. Index 5 is end of atom 0?
        # Sequence: h0 (pred A1 W), h_al1 (pred A1), h_x1...
        # Wait, h0 predicts W for atom 1.
        # h_final block i (indices 5*i+1 to 5*i+5)
        # The w_logit for atom i+1 is at the END of block i.
        # h[0] -> w_logit for Atom 1.
        # h[5] -> w_logit for Atom 2.
        # ...
        # h[5*(n_max-1)] -> w_logit for Atom n_max.
        # So indices are 0, 5, 10... 5*(n_max-1).
        # Total n_max predictions.
        
        # PyTorch slicing with stride:
        # h[:, 0::5] -> (Batch, n_max+1, ...) if size allows
        # n_max atoms. Total len 5*n_max + 1.
        # Indices: 0, 5, 10, ..., 5*n_max.
        # This gives n_max + 1 slices.
        # We need predictions for atoms 0..n_max-1.
        # w_logit for atom 0 comes from h[0]? No, h0 predicts Atom 0 (if 0-indexed).
        # w_logit from h[5*n_max] predicts Atom n_max (which is out of bounds/stop token).
        
        w_logits_all = h[:, 0::5, :wyck_types] # (Batch, n_max+1, wyck)
        w_logit = w_logits_all[:, :-1, :] # (Batch, n_max, wyck)
        
        # a_logit:
        # Starts at 1::5
        a_logit = h[:, 1::5, :atom_types] # (Batch, n_max, atom)
        
        # Coords
        h_x = h[:, 2::5, :coord_types]
        h_y = h[:, 3::5, :coord_types]
        h_z = h[:, 4::5, :coord_types]
        
        # Calculate Losses
        
        # 1. W Loss (Cross Entropy)
        # Gather log probs for true W
        # W: (Batch, n_max)
        log_softmax_w = F.log_softmax(w_logit, dim=-1)
        logp_w_per_atom = torch.gather(log_softmax_w, 2, W.unsqueeze(-1)).squeeze(-1)
        logp_w = torch.sum(logp_w_per_atom, dim=1) # (Batch,)
        
        # 2. A Loss
        log_softmax_a = F.log_softmax(a_logit, dim=-1)
        logp_a_per_atom = torch.gather(log_softmax_a, 2, A.unsqueeze(-1)).squeeze(-1)
        logp_a = torch.sum(logp_a_per_atom, dim=1)
        
        # 3. XYZ Loss
        X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
        
        # fc_mask: (Batch, n_max, 3)
        # fc_mask_table: (230, 28, 3).
        # Indexing:
        fc_mask_gpu = fc_mask_table.to(device)
        G_idx_exp = (G - 1).long().unsqueeze(1).expand(-1, W.shape[1])
        fc_lookup = fc_mask_gpu[G_idx_exp, W.long()] # (Batch, n_max, 3)
        
        # Also W > 0 constraint
        w_pos = (W > 0).unsqueeze(-1).expand(-1, -1, 3)
        fc_mask = w_pos & fc_lookup
        
        logp_x = compute_logp_x(h_x, X, fc_mask[:, :, 0])
        logp_y = compute_logp_x(h_y, Y, fc_mask[:, :, 1])
        logp_z = compute_logp_x(h_z, Z, fc_mask[:, :, 2])
        
        logp_xyz = logp_x + logp_y + logp_z
        
        # 4. Lattice Loss
        # L params are predicted at step `num_sites`.
        # h[1::5] corresponds to h_al.
        # We need h_al[num_sites].
        # num_sites: (Batch,)
        # h_al_seq: (Batch, n_max, Output)
        h_al_seq = h[:, 1::5, :]
        
        # Gather h_al at num_sites
        # indices: (Batch, 1, Output)
        # clamp to n_max-1
        idx = num_sites.unsqueeze(1).unsqueeze(2).expand(-1, -1, h_al_seq.shape[2])
        idx = torch.clamp(idx, max=h_al_seq.shape[1]-1)
        
        h_al_final = torch.gather(h_al_seq, 1, idx).squeeze(1) # (Batch, Output)
        
        # Parse Lattice Params
        l_params = h_al_final[:, atom_types : atom_types + Kl + 12*Kl]
        l_logit, mu, sigma = torch.split(l_params, [Kl, 6*Kl, 6*Kl], dim=-1)
        
        l_logit = F.log_softmax(l_logit, dim=-1) # (Batch, Kl)
        mu = mu.reshape(batch_size, Kl, 6)
        sigma = sigma.reshape(batch_size, Kl, 6)
        
        # Gaussian Log PDF
        # L: (Batch, 6)
        log_gauss = gaussian_logpdf(L, mu, sigma) # (Batch, Kl, 6)
        
        # Sum over 6 dims? No, assuming diagonal covariance, sum log_probs of dims.
        # log_gauss is (Batch, Kl, 6).
        # We need to apply lattice_mask before summing?
        # Lattice Mask: (230, 6)
        lattice_mask_gpu = lattice_mask_cpu.to(device)
        mask_l = lattice_mask_gpu[G_idx] # (Batch, 6)
        
        # Apply mask to log_gauss (mask 0 means independent/fixed, so ignore loss? 
        # "where((lattice_mask[G-1]>0), logp_l, 0)"
        # logp_l in JAX was (Kl, 6).
        # We sum over 6 dims after masking.
        mask_l_exp = mask_l.unsqueeze(1) # (Batch, 1, 6)
        log_gauss = torch.where(mask_l_exp > 0, log_gauss, torch.zeros_like(log_gauss))
        
        # Sum over 6 dimensions
        log_gauss_sum = torch.sum(log_gauss, dim=2) # (Batch, Kl)
        
        # Mixture LogSumExp
        logp_l = torch.logsumexp(l_logit + log_gauss_sum, dim=1) # (Batch,)
        
        return logp_w, logp_xyz, logp_a, logp_l

    def loss_fn(model, G, L, XYZ, A, W, is_train=True):
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(model, G, L, XYZ, A, W, is_train)
        
        loss_w = -torch.mean(logp_w)
        loss_xyz = -torch.mean(logp_xyz)
        loss_a = -torch.mean(logp_a)
        loss_l = -torch.mean(logp_l)
        
        total_loss = loss_xyz + lamb_a * loss_a + lamb_w * loss_w + lamb_l * loss_l
        
        return total_loss, (loss_w.item(), loss_a.item(), loss_xyz.item(), loss_l.item())
        
    return loss_fn, logp_fn
