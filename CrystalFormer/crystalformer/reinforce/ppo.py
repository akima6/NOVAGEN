import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import numpy as np
import copy

# Assumes lattice.py is rewritten and available
from crystalformer.src.lattice import norm_lattice

def make_ppo_loss_fn(logp_fn, eps_clip, beta=0.1):
    KL_MAX = 0.05
    
    def ppo_loss_fn(model, x, old_logp, pretrain_logp, advantages):
        """
        PPO clipped objective function with KL divergence regularization
        """
        # Unpack inputs
        G, L, XYZ, A, W = x
        
        # 1. Compute current log probs (THIS WAS THE MISSING STEP)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(model, G, L, XYZ, A, W, is_train=True)
        new_logp = logp_w + logp_xyz + logp_a + logp_l

        # 2. KL Divergence Penalty
        kl_raw = pretrain_logp - new_logp
        
        # Clamp the raw KL (ensures a safe learning step)
        kl_clamped_mean = torch.mean(torch.clamp(kl_raw, min=-1.0, max=KL_MAX))
        
        # PPO Loss is calculated correctly using the clip term (surr1, surr2)
        kl_penalty = beta * kl_clamped_mean
        
        # 3. Ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(new_logp - old_logp)

        # 4. Surrogate Objectives
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        
        # Final PPO Loss (Minimize)
        ppo_objective = torch.min(surr1, surr2)
        ppo_loss = -torch.mean(ppo_objective) + kl_penalty 
        
        # Return loss and the *clamped* KL for logging
        return ppo_loss, kl_clamped_mean
    
    return ppo_loss_fn

def train(model, optimizer, spg_mask, loss_fn, logp_fn, batch_reward_fn, sample_crystal_fn, 
          epochs, ppo_epochs, batchsize, valid_data, path, checkpoint_interval=5):
    """
    PPO Training Loop.
    
    Args:
        model: PyTorch CrystalTransformer
        optimizer: PyTorch Optimizer
        spg_mask: List/Array of probabilities for Space Groups (1-230)
        loss_fn: Validation loss function (NLL)
        logp_fn: Function to compute log-probs
        batch_reward_fn: Function taking batch x -> rewards
        sample_crystal_fn: Function(model, G) -> (XYZ, A, W, M, L)
        epochs: Total iterations
        ppo_epochs: Number of PPO updates per batch
        batchsize: Batch size
        valid_data: Validation dataset tuple
        path: Output path
    """
    
    device = next(model.parameters()).device
    print(f"PPO Training on device: {device}")

    # Initialize Logging
    log_filename = os.path.join(path, "data.txt")
    if not os.path.exists(log_filename):
        with open(log_filename, "w") as f:
            f.write("epoch f_mean f_err v_loss v_loss_w v_loss_a v_loss_xyz v_loss_l\n")

    # Initialize Baseline (Exponential Moving Average)
    baseline = 0.0
    
    # Create Frozen Pretrain Model (Reference)
    # This replaces 'pretrain_params' from JAX
    pretrain_model = copy.deepcopy(model)
    pretrain_model.eval()
    for p in pretrain_model.parameters():
        p.requires_grad = False
    
    # Define PPO Loss (using default parameters from original file usually 0.2, 0.1)
    # The original file passed eps_clip and beta to make_ppo_loss_fn inside main scripts usually.
    # Here we instantiate it with standard defaults if not passed, but let's assume standard logic.
    ppo_calc = make_ppo_loss_fn(logp_fn, eps_clip=0.2, beta=0.1)
    
    for epoch in range(1, epochs + 1):
        
        # --- 1. SAMPLING ---
        # Sample Space Groups based on mask
        if not isinstance(spg_mask, torch.Tensor):
            spg_mask = torch.tensor(spg_mask, device=device, dtype=torch.float32)
        else:
            spg_mask = spg_mask.to(device).float()
            
        # Normalize mask
        if spg_mask.sum() == 0:
            spg_mask = torch.ones_like(spg_mask)
        spg_probs = spg_mask / spg_mask.sum()
        
        # Sample G (0-229 indices) -> +1 for SG Number
        G_idx = torch.multinomial(spg_probs, batchsize, replacement=True)
        G = (G_idx + 1).long() # (Batch,)
        
        # Sample Crystals
        # We assume sample_crystal_fn binds config args and just accepts (model, G)
        with torch.no_grad():
            XYZ, A, W, M, L = sample_crystal_fn(model, G)
        
        # Generated Batch
        x_gen = (G, L, XYZ, A, W)
        
        # --- 2. REWARD COMPUTATION ---
        # batch_reward_fn typically returns -Energy (so higher is better)
        # Expects x tuple. Returns numpy array or tensor.
        rewards_val = -batch_reward_fn(x_gen)
        
        if isinstance(rewards_val, torch.Tensor):
            rewards = rewards_val.to(device).float()
        else:
            rewards = torch.tensor(rewards_val, device=device, dtype=torch.float32)
            
        f_mean = rewards.mean().item()
        f_err = rewards.std().item() / math.sqrt(batchsize)
        
        # Update Baseline
        if epoch == 1:
            baseline = f_mean
        else:
            baseline = 0.95 * baseline + 0.05 * f_mean
            
        advantages = rewards - baseline
        
        # --- 3. PREPARE DATA ---
        # Normalize lattice for model input
        L_norm = norm_lattice(G, W, L)
        x_train = (G, L_norm, XYZ, A, W)
        
        # --- 4. COMPUTE OLD LOG PROBS ---
        # We need log probs from the current model (before update) and pretrain model
        with torch.no_grad():
            # Old Policy (Current weights)
            lp_w, lp_xyz, lp_a, lp_l = logp_fn(model, G, L_norm, XYZ, A, W, is_train=False)
            old_logp = lp_w + lp_xyz + lp_a + lp_l
            
            # Pretrain Policy (Frozen weights)
            pp_w, pp_xyz, pp_a, pp_l = logp_fn(pretrain_model, G, L_norm, XYZ, A, W, is_train=False)
            pretrain_logp = pp_w + pp_xyz + pp_a + pp_l
            
        # --- 5. PPO UPDATES ---
        model.train()
        avg_ppo_loss = 0.0
        avg_kl = 0.0
        
        for _ in range(ppo_epochs):
            optimizer.zero_grad()
            
            loss, kl = ppo_calc(model, x_train, old_logp, pretrain_logp, advantages)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_ppo_loss += loss.item()
            avg_kl += kl.item()
            
        avg_ppo_loss /= ppo_epochs
        avg_kl /= ppo_epochs
        
        print(f"Epoch {epoch}: Reward={f_mean:.4f} | PPO Loss={avg_ppo_loss:.4f} | KL={avg_kl:.4f}")

        # --- 6. VALIDATION ---
        model.eval()
        valid_loss = 0.0
        v_aux = np.zeros(4)
        
        # Validation Data Unpack
        v_G, v_L, v_XYZ, v_A, v_W = valid_data
        num_val = v_G.shape[0]
        num_batches_val = math.ceil(num_val / batchsize)
        
        with torch.no_grad():
            for i in range(num_batches_val):
                # Batch Slice
                sl = slice(i*batchsize, min((i+1)*batchsize, num_val))
                batch_G = v_G[sl].to(device)
                batch_L = v_L[sl].to(device)
                batch_XYZ = v_XYZ[sl].to(device)
                batch_A = v_A[sl].to(device)
                batch_W = v_W[sl].to(device)
                
                # loss_fn returns total_loss, (aux...)
                val_l, val_a = loss_fn(model, batch_G, batch_L, batch_XYZ, batch_A, batch_W, is_train=False)
                
                valid_loss += val_l.item()
                v_aux += np.array(val_a)
        
        if num_batches_val > 0:
            valid_loss /= num_batches_val
            v_aux /= num_batches_val
            
        # Log to file
        with open(log_filename, "a") as f:
            f.write(("%6d" + 2*"  %.6f" + 5*"  %.6f" + "\n") % (
                epoch, f_mean, f_err, valid_loss, *v_aux
            ))
            
        # --- 7. CHECKPOINT ---
        if epoch % checkpoint_interval == 0:
            ckpt_path = os.path.join(path, f"epoch_{epoch:06d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline': baseline
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    return model, optimizer
