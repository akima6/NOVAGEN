import torch
import torch.nn.functional as F
import numpy as np

# We assume these tables are available from the rewritten wyckoff.py
from crystalformer.src.wyckoff import mult_table, symops
from crystalformer.src.lattice import symmetrize_lattice

def sample_von_mises(loc, kappa, shape):
    """
    Generate sample from von Mises distribution.
    """
    if not isinstance(loc, torch.Tensor):
        loc = torch.tensor(loc)
    
    if not isinstance(kappa, torch.Tensor):
        concentration = torch.tensor(kappa)
    else:
        concentration = kappa
        
    device = loc.device
    if concentration.device != device:
        concentration = concentration.to(device)
        
    concentration = torch.clamp(concentration, min=1e-6)
    
    if loc.numel() == 1 and concentration.numel() == 1:
         loc = loc.expand(shape)
         concentration = concentration.expand(shape)
    
    import torch.distributions as dist
    vm = dist.von_mises.VonMises(loc, concentration)
    
    samples = vm.sample()
    samples = (samples + np.pi) % (2.0 * np.pi) - np.pi
    
    return samples

def project_xyz_batch(G, W, X, idx=0):
    batch_size = G.shape[0]
    device = X.device
    symops_gpu = symops.to(device)
    ops = symops_gpu[G-1, W, idx] 
    ones = torch.ones((batch_size, 1), device=device)
    affine_points = torch.cat([X, ones], dim=1).unsqueeze(2) 
    x_new = torch.bmm(ops, affine_points).squeeze(2)
    x_new -= torch.floor(x_new)
    return x_new

def inference(model, g, W, A, X, Y, Z, is_train=False):
    XYZ = torch.stack([X, Y, Z], dim=-1) 
    device = W.device
    mult_table_gpu = mult_table.to(device)
    G_expanded = (g - 1).unsqueeze(1).expand(-1, W.shape[1])
    M = mult_table_gpu[G_expanded, W]
    return model(g, XYZ, A, W, M, is_train)

def sample_top_p(logits, p, temperature):
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -1e10)
    probs = F.softmax(logits / temperature, dim=1)
    samples = torch.multinomial(probs, 1).squeeze(1) 
    return samples

def sample_x_fn(h_x, Kx, top_p, temperature, batchsize):
    coord_types = 3 * Kx
    relevant = h_x[:, :coord_types]
    x_logit, loc, kappa = torch.split(relevant, [Kx, Kx, Kx], dim=-1)
    k = sample_top_p(x_logit, top_p, temperature) 
    k_uns = k.unsqueeze(1)
    selected_loc = torch.gather(loc, 1, k_uns).squeeze(1)
    selected_kappa = torch.gather(kappa, 1, k_uns).squeeze(1)
    x = sample_von_mises(selected_loc, selected_kappa / temperature, (batchsize,))
    x = (x + np.pi) / (2.0 * np.pi)
    return x

def sample_crystal(key, transformer, params, n_max, batchsize, atom_types, wyck_types, Kx, Kl, g, w_mask, atom_mask, top_p, temperature, T1, constraints):
    device = next(transformer.parameters()).device
    
    if isinstance(g, int):
        G = torch.full((batchsize,), g, device=device).long()
    else:
        G = g.to(device)
        
    W = torch.zeros((batchsize, n_max), dtype=torch.long, device=device)
    A = torch.zeros((batchsize, n_max), dtype=torch.long, device=device)
    X = torch.zeros((batchsize, n_max), device=device)
    Y = torch.zeros((batchsize, n_max), device=device)
    Z = torch.zeros((batchsize, n_max), device=device)
    L = torch.zeros((batchsize, n_max, Kl + 12 * Kl), device=device)
    
    transformer.eval() 

    # Autoregressive Loop
    for i in range(n_max):
        output = inference(transformer, G, W, A, X, Y, Z)
        
        # 1. Sample W
        w_logit = output[:, 5 * i, :wyck_types]
        w = sample_top_p(w_logit, top_p, temperature)
        if w_mask is not None and w_mask[i] is not None:
            w[:] = w_mask[i]
        W[:, i] = w

        # 2. Sample A
        output = inference(transformer, G, W, A, X, Y, Z)
        h_al = output[:, 5 * i + 1]
        a_logit = h_al[:, :atom_types]
        if atom_mask is not None:
            mask_i = atom_mask[i].to(device).unsqueeze(0)
            a_logit = a_logit + torch.where(mask_i > 0, torch.tensor(0.0, device=device), torch.tensor(-1e10, device=device))
        curr_temp = T1 if i == 0 else temperature
        a = sample_top_p(a_logit, top_p, curr_temp)
        if constraints is not None and constraints[i] < i:
            ref_idx = constraints[i]
            a = A[:, ref_idx]
        A[:, i] = a
        
        l_params = h_al[:, atom_types : atom_types + Kl + 12 * Kl]
        L[:, i] = l_params
        
        # 3. Sample X
        output = inference(transformer, G, W, A, X, Y, Z)
        h_x = output[:, 5 * i + 2]
        x = sample_x_fn(h_x, Kx, top_p, temperature, batchsize)
        xyz_temp = torch.stack([x, torch.zeros(batchsize, device=device), torch.zeros(batchsize, device=device)], dim=1)
        xyz_proj = project_xyz_batch(G, w, xyz_temp, idx=0)
        x = xyz_proj[:, 0]
        X[:, i] = x
        
        # 4. Sample Y
        output = inference(transformer, G, W, A, X, Y, Z)
        h_y = output[:, 5 * i + 3]
        y = sample_x_fn(h_y, Kx, top_p, temperature, batchsize)
        xyz_temp = torch.stack([X[:, i], y, torch.zeros(batchsize, device=device)], dim=1)
        xyz_proj = project_xyz_batch(G, w, xyz_temp, idx=0)
        y = xyz_proj[:, 1]
        Y[:, i] = y
        
        # 5. Sample Z
        output = inference(transformer, G, W, A, X, Y, Z)
        h_z = output[:, 5 * i + 4]
        z = sample_x_fn(h_z, Kx, top_p, temperature, batchsize)
        xyz_temp = torch.stack([X[:, i], Y[:, i], z], dim=1)
        xyz_proj = project_xyz_batch(G, w, xyz_temp, idx=0)
        z = xyz_proj[:, 2]
        Z[:, i] = z
        
    # --- Process Lattice ---
    num_sites = torch.sum(A != 0, dim=1)
    num_sites_uns = num_sites.unsqueeze(1).unsqueeze(2).expand(-1, -1, L.shape[2])
    num_sites_uns = torch.clamp(num_sites_uns, max=n_max-1)
    
    L_selected = torch.gather(L, 1, num_sites_uns).squeeze(1)
    l_logit, mu, sigma = torch.split(L_selected, [Kl, 6*Kl, 6*Kl], dim=-1)
    k = sample_top_p(l_logit, top_p, temperature)
    
    mu = mu.reshape(batchsize, Kl, 6)
    sigma = sigma.reshape(batchsize, Kl, 6)
    
    k_uns = k.unsqueeze(1).unsqueeze(2).expand(-1, -1, 6)
    mu_sel = torch.gather(mu, 1, k_uns).squeeze(1)
    sigma_sel = torch.gather(sigma, 1, k_uns).squeeze(1)
    
    L_final = torch.normal(mean=mu_sel, std=sigma_sel * np.sqrt(temperature))
    
    # Force Positive
    lengths = torch.abs(L_final[:, :3]) + 0.1 
    angles = L_final[:, 3:]
    L_final = torch.cat([lengths, angles], dim=-1)
    
    mult_table_gpu = mult_table.to(device)
    G_idx = (G - 1).unsqueeze(1).expand(-1, n_max)
    M = mult_table_gpu[G_idx, W]
    num_atoms = torch.sum(M, dim=1).float()
    
    length, angle = torch.split(L_final, 3, dim=-1)
    
    # --- PHYSICS PATCH: SCALING FACTOR ---
    # Fix for Low Density (0.4 -> 2.0 g/cm3)
    CORRECTION_FACTOR = 0.6 
    length = length * torch.pow(num_atoms.unsqueeze(1), 1.0/3.0) * CORRECTION_FACTOR
    
    angle = angle * (180.0 / np.pi)
    
    L_final = torch.cat([length, angle], dim=-1)
    L_final = symmetrize_lattice(G, L_final)
    
    XYZ_final = torch.stack([X, Y, Z], dim=-1)
    
    return XYZ_final, A, W, M, L_final
