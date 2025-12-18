import torch
import pickle
import numpy as np
import os

# CONFIGURATION
PKL_PATH = r"pretrained_model\epoch_005500.pkl"
PT_PATH  = r"pretrained_model\epoch_005500_CLEAN.pt"

def flatten_dict(d, parent_key='', sep='/'):
    """
    Recursively flattens a nested dictionary into a single level 
    dict with keys separated by 'sep'.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def to_torch(jax_arr, name=""):
    """Convert numpy/jax array to torch tensor and transpose if needed."""
    tensor = torch.from_numpy(np.array(jax_arr))
    # Transpose 2D weights (Linear layers), but ignore Embeddings (Vocab, Dim)
    if len(tensor.shape) == 2:
        if 'embedding' in name.lower() or 'emb' in name.lower():
            pass 
        else:
            tensor = tensor.T 
    return tensor

def convert():
    print(f"Loading {PKL_PATH}...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Handle the 'params' wrapper if it exists
    raw_params = data.get('params', data)
    
    print("Flattening dictionary structure...")
    # This ensures we can access everything via "layer/sublayer/param" keys
    flat_params = flatten_dict(raw_params)
    
    torch_state_dict = {}
    print(f"Total flattened keys found: {len(flat_params)}")

    # --- 1. EMBEDDINGS ---
    # Using the exact keys from your inspect_checkpoint output
    # Note: inspect output showed '~/g_embeddings', so we use that key directly.
    
    torch_state_dict['g_embeddings.weight'] = to_torch(flat_params['~/g_embeddings'], 'emb')
    torch_state_dict['w_embeddings.weight'] = to_torch(flat_params['~/w_embeddings'], 'emb')
    torch_state_dict['a_embeddings.weight'] = to_torch(flat_params['~/a_embeddings'], 'emb')

    # --- 2. INPUT PROJECTIONS ---
    # linear, linear_1 ... linear_6
    torch_state_dict['h0_mlp.0.weight'] = to_torch(flat_params['linear/w'], 'linear')
    torch_state_dict['h0_mlp.0.bias']   = to_torch(flat_params['linear/b'], 'linear')
    
    torch_state_dict['h0_mlp.2.weight'] = to_torch(flat_params['linear_1/w'], 'linear')
    torch_state_dict['h0_mlp.2.bias']   = to_torch(flat_params['linear_1/b'], 'linear')

    torch_state_dict['fc_hW.weight'] = to_torch(flat_params['linear_2/w'], 'linear')
    torch_state_dict['fc_hW.bias']   = to_torch(flat_params['linear_2/b'], 'linear')
    
    torch_state_dict['fc_hA.weight'] = to_torch(flat_params['linear_3/w'], 'linear')
    torch_state_dict['fc_hA.bias']   = to_torch(flat_params['linear_3/b'], 'linear')

    torch_state_dict['fc_hX.weight'] = to_torch(flat_params['linear_4/w'], 'linear')
    torch_state_dict['fc_hX.bias']   = to_torch(flat_params['linear_4/b'], 'linear')
    
    torch_state_dict['fc_hY.weight'] = to_torch(flat_params['linear_5/w'], 'linear')
    torch_state_dict['fc_hY.bias']   = to_torch(flat_params['linear_5/b'], 'linear')
    
    torch_state_dict['fc_hZ.weight'] = to_torch(flat_params['linear_6/w'], 'linear')
    torch_state_dict['fc_hZ.bias']   = to_torch(flat_params['linear_6/b'], 'linear')

    # --- 3. TRANSFORMER LAYERS (0 to 15) ---
    for i in range(16):
        # ATTENTION NAMES
        # Layer 0 is 'multi_head_attention', Layer 1 is 'multi_head_attention_1'
        jax_attn = "multi_head_attention" if i == 0 else f"multi_head_attention_{i}"
        pt_attn  = f"layers.{i}.attn"
        
        # We construct the full flat key manually
        torch_state_dict[f"{pt_attn}.q_proj.weight"] = to_torch(flat_params[f"{jax_attn}/query/w"], 'linear')
        torch_state_dict[f"{pt_attn}.q_proj.bias"]   = to_torch(flat_params[f"{jax_attn}/query/b"], 'linear')
        
        torch_state_dict[f"{pt_attn}.k_proj.weight"] = to_torch(flat_params[f"{jax_attn}/key/w"], 'linear')
        torch_state_dict[f"{pt_attn}.k_proj.bias"]   = to_torch(flat_params[f"{jax_attn}/key/b"], 'linear')
        
        torch_state_dict[f"{pt_attn}.v_proj.weight"] = to_torch(flat_params[f"{jax_attn}/value/w"], 'linear')
        torch_state_dict[f"{pt_attn}.v_proj.bias"]   = to_torch(flat_params[f"{jax_attn}/value/b"], 'linear')
        
        torch_state_dict[f"{pt_attn}.o_proj.weight"] = to_torch(flat_params[f"{jax_attn}/linear/w"], 'linear')
        torch_state_dict[f"{pt_attn}.o_proj.bias"]   = to_torch(flat_params[f"{jax_attn}/linear/b"], 'linear')

        # MLP NAMES
        idx_expand = 7 + (2 * i)
        idx_contract = 8 + (2 * i)
        
        torch_state_dict[f"layers.{i}.mlp.0.weight"] = to_torch(flat_params[f"linear_{idx_expand}/w"], 'linear')
        torch_state_dict[f"layers.{i}.mlp.0.bias"]   = to_torch(flat_params[f"linear_{idx_expand}/b"], 'linear')
        
        torch_state_dict[f"layers.{i}.mlp.2.weight"] = to_torch(flat_params[f"linear_{idx_contract}/w"], 'linear')
        torch_state_dict[f"layers.{i}.mlp.2.bias"]   = to_torch(flat_params[f"linear_{idx_contract}/b"], 'linear')

        # LAYER NORM NAMES
        idx_ln1 = 2 * i
        idx_ln2 = 2 * i + 1
        name_ln1 = "layer_norm" if idx_ln1 == 0 else f"layer_norm_{idx_ln1}"
        name_ln2 = f"layer_norm_{idx_ln2}"
        
        torch_state_dict[f"layers.{i}.ln1.weight"] = to_torch(flat_params[f"{name_ln1}/scale"])
        torch_state_dict[f"layers.{i}.ln1.bias"]   = to_torch(flat_params[f"{name_ln1}/offset"])
        
        torch_state_dict[f"layers.{i}.ln2.weight"] = to_torch(flat_params[f"{name_ln2}/scale"])
        torch_state_dict[f"layers.{i}.ln2.bias"]   = to_torch(flat_params[f"{name_ln2}/offset"])

    # --- 4. FINAL OUTPUT ---
    torch_state_dict['final_norm.weight'] = to_torch(flat_params['layer_norm_32/scale'])
    torch_state_dict['final_norm.bias']   = to_torch(flat_params['layer_norm_32/offset'])

    torch_state_dict['output_proj.weight'] = to_torch(flat_params['linear_39/w'], 'linear')
    torch_state_dict['output_proj.bias']   = to_torch(flat_params['linear_39/b'], 'linear')

    print(f"Conversion complete. Saving to {PT_PATH}...")
    torch.save(torch_state_dict, PT_PATH)
    print("DONE. You can now use this .pt file in your PyTorch model.")

if __name__ == "__main__":
    if not os.path.exists(PKL_PATH):
        print(f"Error: Could not find {PKL_PATH}")
    else:
        convert()