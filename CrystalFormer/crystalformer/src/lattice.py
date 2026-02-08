import torch
import numpy as np
from crystalformer.src.wyckoff import mult_table

def make_lattice_mask():
    '''
    return mask for independent lattice params 
    '''
    # 1-2
    # 3-15 
    # 16-74
    # 75-142
    # 143-194
    # 195-230    
    mask = [1, 1, 1, 1, 1, 1] * 2 +\
           [1, 1, 1, 0, 1, 0] * 13+\
           [1, 1, 1, 0, 0, 0] * 59+\
           [1, 0, 1, 0, 0, 0] * 68+\
           [1, 0, 1, 0, 0, 0] * 52+\
           [1, 0, 0, 0, 0, 0] * 36

    # Convert to tensor. Note: Device will be CPU by default, moved when needed.
    return torch.tensor(mask, dtype=torch.int32).reshape(230, 6)

def symmetrize_lattice(spacegroup, lattice):
    '''
    Place lattice params into lattice according to the space group constraints.
    Args:
        spacegroup: (Batch,) or scalar Int Tensor
        lattice: (Batch, 6) Float Tensor [a, b, c, alpha, beta, gamma]
    Returns:
        L: (Batch, 6) Symmetrized lattice
    '''
    # Ensure inputs are tensors
    if not isinstance(spacegroup, torch.Tensor):
        spacegroup = torch.tensor(spacegroup)
    if not isinstance(lattice, torch.Tensor):
        lattice = torch.tensor(lattice)
        
    device = lattice.device
    spacegroup = spacegroup.to(device)

    # Handle single sample case by unsqueezing
    is_batch = lattice.ndim == 2
    if not is_batch:
        lattice = lattice.unsqueeze(0)
        spacegroup = spacegroup.reshape(1)

    a, b, c, alpha, beta, gamma = lattice.unbind(-1)
    
    # Constants: Must be same shape as 'a' (Batch size)
    ninety = torch.full_like(a, 90.0)
    onetwenty = torch.full_like(a, 120.0)

    # Construct candidate lattices
    # Triclinic (1-2): [a, b, c, alpha, beta, gamma] (Input)
    
    # Monoclinic (3-15): [a, b, c, 90, beta, 90]
    L_mono = torch.stack([a, b, c, ninety, beta, ninety], dim=-1)
    
    # Orthorhombic (16-74): [a, b, c, 90, 90, 90]
    L_ortho = torch.stack([a, b, c, ninety, ninety, ninety], dim=-1)
    
    # Tetragonal (75-142): [a, a, c, 90, 90, 90]
    L_tetra = torch.stack([a, a, c, ninety, ninety, ninety], dim=-1)
    
    # Hexagonal/Trigonal (143-194): [a, a, c, 90, 90, 120]
    L_hex = torch.stack([a, a, c, ninety, ninety, onetwenty], dim=-1)
    
    # Cubic (195-230): [a, a, a, 90, 90, 90]
    L_cubic = torch.stack([a, a, a, ninety, ninety, ninety], dim=-1)

    # Apply constraints hierarchically using torch.where
    # Logic:
    # If SG <= 2: Keep Original. Else: Check next level.
    # If SG <= 15: Use Mono. Else: Check next level.
    # ...
    
    # Broadcasting spacegroup for comparison
    sg = spacegroup.unsqueeze(-1) # (Batch, 1)

    L = torch.where(sg <= 2, lattice, L_mono)
    L = torch.where(sg <= 15, L, L_ortho)
    L = torch.where(sg <= 74, L, L_tetra)
    L = torch.where(sg <= 142, L, L_hex)
    L = torch.where(sg <= 194, L, L_cubic)
    
    if not is_batch:
        return L.squeeze(0)
    return L


def norm_lattice(G, W, L):
    """
    normalize the lattice lengths by the number of atoms in the unit cell,
    change the lattice angles to radian.
    a -> a/n_atoms^(1/3)
    angle -> angle * pi/180
    
    Args:
        G: (Batch,)
        W: (Batch, n_max)
        L: (Batch, 6)
    """
    # Mult table lookup
    # mult_table is (230, 28)
    # Need to match device
    device = L.device
    mult_table_gpu = mult_table.to(device)
    
    # Adjust G to 0-index
    G_idx = (G - 1).long()
    
    # Gather multiplicity
    # G_idx: (Batch,) -> (Batch, 1) -> (Batch, n_max)
    # W: (Batch, n_max)
    # Check dimensions
    if G_idx.ndim == 1:
        G_expanded = G_idx.unsqueeze(1).expand(-1, W.shape[1])
    else:
        G_expanded = G_idx

    M = mult_table_gpu[G_expanded, W.long()] # (Batch, n_max)
    
    num_atoms = torch.sum(M, dim=1).float() # (Batch,)
    
    # Split L
    length, angle = torch.split(L, 3, dim=-1)
    
    # Normalize length
    # num_atoms^(1/3) needs to be broadcast to (Batch, 3)
    # Handle num_atoms = 0 safely
    num_atoms = torch.clamp(num_atoms, min=1.0)
    
    scale = torch.pow(num_atoms, 1.0/3.0).unsqueeze(1)
    length = length / scale
    
    # Normalize angle (deg -> rad)
    angle = angle * (np.pi / 180.0)
    
    L_norm = torch.cat([length, angle], dim=-1)
    
    return L_norm

if __name__ == '__main__':
    mask = make_lattice_mask()
    print (mask)

    # Test logic
    lattice = torch.randn(1, 6).repeat(3, 1)
    G = torch.tensor([25, 99, 221]) # Ortho, Tetra, Cubic
    
    L = symmetrize_lattice(G, lattice)
    print (L)
