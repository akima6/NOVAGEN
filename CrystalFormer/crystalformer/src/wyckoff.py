import pandas as pd
import os
import numpy as np 
import re
import torch
import sys
def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27
    (Note: 0 is reserved for the pad token)
    """
    letter = letter.lower()
    if 'a' <= letter <= 'z':
        return ord(letter) - ord('a') + 1
    elif letter == 'a' and len(letter) == 1: # Catch for simple 'A' if used
        return 27
    else:
        return 0 # Return 0 (pad) for invalid or unhandled chars
        
def from_xyz_str(xyz_str: str):
    """
    Args:
        xyz_str: string of the form 'x, y, z', '-x, -y, z', '-2y+1/2, 3x+1/2, z-y+1/2', etc.
    Returns:
        affine operator as a 3x4 array
    """
    rot_matrix = np.zeros((3, 3))
    trans = np.zeros(3)
    tokens = xyz_str.strip().replace(" ", "").lower().split(",")
    re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
    re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")
    for i, tok in enumerate(tokens):
        # build the rotation matrix
        for m in re_rot.finditer(tok):
            factor = -1.0 if m.group(1) == "-" else 1.0
            if m.group(2) != "":
                factor *= float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            j = ord(m.group(4)) - 120
            rot_matrix[i, j] = factor
        # build the translation vector
        for m in re_trans.finditer(tok):
            factor = -1 if m.group(1) == "-" else 1
            num = float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            trans[i] = num * factor
    return np.concatenate( [rot_matrix, trans[:, None]], axis=1) # (3, 4)


# Load data
# Assuming the file structure is maintained: ../data/wyckoff_list.csv
try:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/wyckoff_list.csv'))
except FileNotFoundError:
    # Fallback for testing or if file is elsewhere
    # For now we assume strict adherence to provided structure
    print("Warning: ../data/wyckoff_list.csv not found. Please ensure data exists.")
    df = pd.DataFrame(columns=['Wyckoff Positions']) 

if not df.empty:
    df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list
    wyckoff_positions = df['Wyckoff Positions'].tolist()
else:
    wyckoff_positions = []

# Initialize Tables (Numpy for construction)
symops_np = np.zeros((230, 28, 576, 3, 4)) # 576 is the least common multiple for all possible mult
mult_table_np = np.zeros((230, 28), dtype=int) # mult_table[g-1, w] = multiplicity , 28 because we had pad 0 
wmax_table_np = np.zeros((230,), dtype=int)    # wmax_table[g-1] = number of possible wyckoff letters for g 
dof0_table_np = np.ones((230, 28), dtype=bool)  # dof0_table[g-1, w] = True for those wyckoff points with dof = 0 (no continuous dof)
fc_mask_table_np = np.zeros((230, 28, 3), dtype=bool) # fc_mask_table[g-1, w] = True for continuous fc 

def build_g_code():
    #use general wyckoff position as the code for space groups
    xyz_table = []
    g_table = []
    for g in range(230):
        if not wyckoff_positions: break
        wp0 = wyckoff_positions[g][0]
        g_table.append([])
        for xyz in wp0:
            if xyz not in xyz_table: 
                xyz_table.append(xyz)
            g_table[-1].append(xyz_table.index(xyz))
        assert len(g_table[-1]) == len(set(g_table[-1]))

    g_code = []
    for g in range(230):
        g_code.append( [1 if i in g_table[g] else 0 for i in range(len(xyz_table))] )
    
    g_code = torch.tensor(g_code)
    return g_code

# Populate Tables
if wyckoff_positions:
    for g in range(230):
        wyckoffs = []
        for x in wyckoff_positions[g]:
            wyckoffs.append([])
            for y in x:
                wyckoffs[-1].append(from_xyz_str(y))
        wyckoffs = wyckoffs[::-1] # a-z,A

        mult = [len(w) for w in wyckoffs]
        mult_table_np[g, 1:len(mult)+1] = mult
        wmax_table_np[g] = len(mult)

        for w, wyckoff in enumerate(wyckoffs):
            wyckoff = np.array(wyckoff)
            repeats = symops_np.shape[2] // wyckoff.shape[0]
            symops_np[g, w+1, :, :, :] = np.tile(wyckoff, (repeats, 1, 1))
            dof0_table_np[g, w+1] = np.linalg.matrix_rank(wyckoff[0, :3, :3]) == 0
            fc_mask_table_np[g, w+1] = np.abs(wyckoff[0, :3, :3]).sum(axis=1)!=0 

# Convert to Numpy arrays (finalizing structure)
# We keep these as numpy for easy import in other files (transformer.py expects lists/arrays often for indexing)
# But we also provide Tensor versions for the model to use directly on GPU.

symops = torch.from_numpy(symops_np).float()
mult_table = torch.from_numpy(mult_table_np).long()
wmax_table = torch.from_numpy(wmax_table_np).long()
dof0_table = torch.from_numpy(dof0_table_np).bool()
fc_mask_table = torch.from_numpy(fc_mask_table_np).bool()

# Keep numpy versions if needed for simple indexing without tensor overhead in utils
mult_table_numpy = mult_table_np
wmax_table_numpy = wmax_table_np
dof0_table_numpy = dof0_table_np

def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops 
    we need to do that because the sampled atom might not be at the first WP
    Args:
       g: int (space group)
       w: int (wyckoff index)
       x: (3,) Tensor
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''
    # Ensure inputs are tensors on correct device
    device = x.device
    
    # Global tables need to be on the same device
    # We load them lazily or assume they are small enough to copy
    # symops is large (230*28*576*3*4 * 4 bytes ~ 300MB). 
    # Better to index on CPU then move small chunk to GPU.
    
    # 1. Get symmetry operations for general position (w_max)
    g_idx = g - 1
    w_max = wmax_table[g_idx].item()
    m_max = mult_table[g_idx, w_max].item()
    
    # Indexing on CPU is faster than moving whole 300MB table to GPU if not needed
    # But x is on GPU.
    ops_cpu = symops[g_idx, w_max, :m_max] # (m_max, 3, 4)
    ops = ops_cpu.to(device)
    
    affine_point = torch.cat([x, torch.tensor([1.0], device=device)]) # (4, )
    
    # (m,3,4) @ (4,1) -> (m,3)
    # ops is (m, 3, 4). affine_point is (4,)
    # We want ops @ affine_point.
    # torch.mv needs (3,4) and (4,). 
    # Batched: (m,3,4) x (4,) -> (m,3)
    coords = torch.matmul(ops, affine_point) 
    coords -= torch.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # op0 is symops[g-1, w, 0]
    op0_cpu = symops[g_idx, w, 0] # (3, 4)
    op0 = op0_cpu.to(device)
    
    # We want to find coord in `coords` such that dist(op0(coord), coord) is minimized (ideally 0)
    # dist = || (op0 @ [c,1]) - c ||^2 (modulo integer shifts)
    
    # Prepare broadcasting
    # coords: (m_max, 3)
    # op0: (3, 4)
    
    # op0_c = op0 @ [coords, 1]
    # construct [coords, 1] -> (m_max, 4)
    ones = torch.ones((coords.shape[0], 1), device=device)
    coords_affine = torch.cat([coords, ones], dim=1) # (m_max, 4)
    
    # (m_max, 4) @ (4, 3) -> (m_max, 3)
    # op0.T is (4, 3)
    op0_res = torch.matmul(coords_affine, op0.t()) # (m_max, 3)
    
    diff = op0_res - coords
    diff -= torch.round(diff)
    dists = torch.sum(diff**2, dim=1) # (m_max,)
    
    loc = torch.argmin(dists)
    x_gen = coords[loc] # (3,)

    # (3) lastly, apply the given symmetry op to x_gen
    m = mult_table[g_idx, w].item()
    ops_final_cpu = symops[g_idx, w, :m] # (m, 3, 4)
    ops_final = ops_final_cpu.to(device)
    
    affine_point_final = torch.cat([x_gen, torch.tensor([1.0], device=device)])
    xs = torch.matmul(ops_final, affine_point_final) # (m, 3)
    xs -= torch.floor(xs)
    
    return xs

if __name__=='__main__':
    print (symops.shape)
    # Size in MB
    print (symops.numel() * symops.element_size() // (1024*1024))

    np.set_printoptions(threshold=np.inf)

    # Test access (CPU tensor)
    print (symops[166-1, 3, :6])
    op = symops[166-1, 3, 0]
    print (op)
    
    w_max = wmax_table[225-1]
    m_max = mult_table[225-1, w_max]
    print ('w_max, m_max', w_max, m_max)

    print (fc_mask_table[225-1, 6])
    
    # Just printing examples as per original script
    # To exit cleanly in test run
    # sys.exit(0)
