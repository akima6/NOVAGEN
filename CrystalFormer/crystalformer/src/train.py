import torch
import os
import math
import time
from crystalformer.src.utils import shuffle

# We remove the custom checkpoint import as we use torch.save directly
# import crystalformer.src.checkpoint as checkpoint

def to_device(data, device):
    """Helper to move a list of tensors to the target device."""
    return [d.to(device) for d in data]

def train(model, optimizer, loss_fn, start_epoch, epochs, batchsize, train_data, valid_data, path, val_interval):
    """
    PyTorch training loop for CrystalFormer.
    
    Args:
        model: PyTorch nn.Module
        optimizer: PyTorch optimizer instance
        loss_fn: Function with signature (model, G, L, X, A, W, is_train) -> (loss, aux_tuple)
        start_epoch: Integer, last finished epoch (0 for new run)
        epochs: Total number of epochs
        batchsize: Batch size
        train_data: Tuple of tensors (G, L, XYZ, A, W)
        valid_data: Tuple of tensors (G, L, XYZ, A, W)
        path: Directory to save logs and checkpoints
        val_interval: Epoch interval for validation
    """
    
    # Detect device from model parameters
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
        
    print(f"Training on device: {device}")
    print(f"Batch size: {batchsize}")

    # Logging setup
    log_filename = os.path.join(path, "data.txt")
    
    # Check if we are resuming or starting new
    # If starting new (start_epoch == 0), 'w' mode. Else 'a'.
    mode = "w" if start_epoch == 0 else "a"
    
    # Initialize log file header if empty or new
    if not os.path.exists(log_filename) or os.path.getsize(log_filename) == 0:
        with open(log_filename, "w") as f:
            f.write("epoch t_loss v_loss t_loss_w v_loss_w t_loss_a v_loss_a t_loss_xyz v_loss_xyz t_loss_l v_loss_l\n")
 
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        
        # Shuffle train data
        # utils.shuffle now returns tensors
        train_data = shuffle(train_data)

        # Unpack to get number of samples (L is index 1)
        # train_data is (G, L, X, A, W)
        num_samples = train_data[1].shape[0]
        num_batches = math.ceil(num_samples / batchsize)
        
        model.train()
        train_loss = 0.0 
        train_aux = torch.zeros(4, device=device) # w, a, xyz, l

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            
            # Slice batch and move to GPU
            batch = [x[start_idx:end_idx] for x in train_data]
            batch = to_device(batch, device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward + Loss
            # We assume loss_fn calls the model forward pass internally
            # signature: loss_fn(model, G, L, X, A, W, is_train)
            loss, aux = loss_fn(model, *batch, is_train=True)
            
            # Backward
            loss.backward()
            
            # Gradient Clipping (Optional but recommended for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            train_aux += torch.tensor(aux, device=device)

        # Average over batches
        train_loss /= num_batches
        train_aux /= num_batches

        # Validation Loop
        valid_loss = 0.0
        valid_aux = torch.zeros(4, device=device)
        
        if epoch % val_interval == 0:
            model.eval()
            
            num_samples_val = valid_data[1].shape[0]
            num_batches_val = math.ceil(num_samples_val / batchsize)
            
            with torch.no_grad():
                for batch_idx in range(num_batches_val):
                    start_idx = batch_idx * batchsize
                    end_idx = min(start_idx + batchsize, num_samples_val)
                    
                    batch = [x[start_idx:end_idx] for x in valid_data]
                    batch = to_device(batch, device)
                    
                    loss, aux = loss_fn(model, *batch, is_train=False)
                    
                    valid_loss += loss.item()
                    valid_aux += torch.tensor(aux, device=device)
            
            valid_loss /= num_batches_val
            valid_aux /= num_batches_val

        # Unpack aux values for logging
        t_w, t_a, t_xyz, t_l = train_aux.tolist()
        v_w, v_a, v_xyz, v_l = valid_aux.tolist()

        # Write to log
        log_str = ("%6d" + 10*"  %.6f" + "\n") % (epoch, 
                                                train_loss,   valid_loss,
                                                t_w, v_w, 
                                                t_a, v_a, 
                                                t_xyz, v_xyz, 
                                                t_l, v_l
                                                )
        
        with open(log_filename, "a") as f:
            f.write(log_str)

        # Checkpointing
        # We use standard PyTorch saving
        ckpt_filename = os.path.join(path, "epoch_%06d.pt" %(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, ckpt_filename)
        print(f"Epoch {epoch} finished. Saved checkpoint to {ckpt_filename}")
                
    return model, optimizer