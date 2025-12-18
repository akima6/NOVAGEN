import torch
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
import multiprocessing
import os

# We assume these tables will be available from the rewritten wyckoff.py
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.elements import element_list

def sort_atoms(W, A, X):
    """
    Lex sort atoms according W, X, Y, Z.
    Supports batched inputs.
    
    Args:
        W: (Batch, n) or (n,)
        A: (Batch, n) or (n,)
        X: (Batch, n, 3) or (n, 3)
        
    Returns:
        A: Sorted atom types
        X: Sorted coordinates
    """
    # Handle non-batched input by adding a batch dim temporarily
    is_batched = W.ndim == 2
    if not is_batched:
        W = W.unsqueeze(0)
        A = A.unsqueeze(0)
        X = X.unsqueeze(0)

    # W_temp logic: change 0 to 9999 so they remain at the end after sort
    # Original: jnp.where(W>0, W, 9999)
    W_temp = torch.where(W > 0, W, torch.tensor(9999, device=W.device, dtype=W.dtype))

    # X floor logic
    X_floor = X - torch.floor(X)
    
    # Lexsort: JAX/Numpy lexsort((k1, k2, k3, k4)) uses k4 as primary.
    # Keys from original: (X[:,2], X[:,1], X[:,0], W_temp)
    # So W_temp is Primary, X0 Secondary, X1 Tertiary, X2 Quaternary.
    # In PyTorch, we can achieve this by performing stable sorts in REVERSE order of importance:
    # Sort by X2, then X1, then X0, then W_temp.
    
    # We need to sort along the 'n' dimension (dim=1)
    
    # 1. Sort by X[:, :, 2]
    idx = torch.argsort(X_floor[:, :, 2], dim=1, stable=True)
    W_temp = torch.gather(W_temp, 1, idx)
    X_floor = torch.gather(X_floor, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    # We also need to permute the original A and X to keep them in sync for the next sort steps
    # But actually, we just need the final indices. 
    # To do this correctly without re-indexing everything repeatedly, 
    # we can re-order the keys step-by-step.
    
    # Let's restart with a cleaner approach for indices:
    # Initialize indices as 0..n
    batch_size, n = W.shape
    indices = torch.arange(n, device=W.device).unsqueeze(0).expand(batch_size, -1) # (B, n)
    
    keys = [
        X_floor[:, :, 2], # Least significant
        X_floor[:, :, 1],
        X_floor[:, :, 0],
        W_temp            # Most significant
    ]
    
    for k in keys:
        # Gather the current key using the current best permutation
        # k is (B, n). We want k_ordered = k[indices]
        k_ordered = torch.gather(k, 1, indices)
        
        # Sort this key (stable)
        sort_idx = torch.argsort(k_ordered, dim=1, stable=True)
        
        # Update the global indices
        indices = torch.gather(indices, 1, sort_idx)

    # Apply final indices to A and X
    # A: (B, n)
    A_sorted = torch.gather(A, 1, indices)
    
    # X: (B, n, 3)
    # Expand indices to (B, n, 3)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
    X_sorted = torch.gather(X, 1, indices_expanded)

    if not is_batched:
        return A_sorted.squeeze(0), X_sorted.squeeze(0)
    
    return A_sorted, X_sorted

def letter_to_number(letter):
    """
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    """
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def shuffle(data):
    """
    shuffle data along batch dimension
    Args:
        data: tuple of tensors (G, L, XYZ, A, W)
    """
    G, L, XYZ, A, W = data
    batch_size = len(L)
    idx = torch.randperm(batch_size)
    return G[idx], L[idx], XYZ[idx], A[idx], W[idx]
    
def process_one(cif, atom_types, wyck_types, n_max, tol=0.01):
    """
    Process one cif string to get G, L, XYZ, A, W
    (Kept largely original as it uses CPU-bound Pymatgen/PyXtal)
    """
    try: crystal = Structure.from_str(cif, fmt='cif')
    except: crystal = Structure.from_dict(eval(cif))
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    
    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites) 

    # print (g, c.group.symbol, num_sites)
    natoms = 0
    ww = []
    aa = []
    fc = []
    ws = []
    for site in c.atom_sites:
        a = element_list.index(site.specie) 
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        # print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    # print (ws, aa, ww, natoms) 

    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    
    # print ('===================================')

    return g, l, fc, aa, ww 

def GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, num_workers=1):
    """
    Read cif strings from csv file and convert them to G, L, XYZ, A, W
    Returns PyTorch Tensors.
    """
    if csv_file.endswith('.lmdb'):
        import lmdb
        import pickle
        # read from lmdb
        env = lmdb.open(
            csv_file,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        contents = env.begin().cursor().iternext()
        data = tuple([pickle.loads(value) for _, value in contents])
        G, L, XYZ, A, W = data
        print('G:', G.shape)
        print('L:', L.shape)
        print('XYZ:', XYZ.shape)
        print('A:', A.shape)
        print('W:', W.shape)
        # Convert to torch
        G = torch.from_numpy(G) if isinstance(G, np.ndarray) else torch.tensor(G)
        L = torch.from_numpy(L) if isinstance(L, np.ndarray) else torch.tensor(L)
        XYZ = torch.from_numpy(XYZ) if isinstance(XYZ, np.ndarray) else torch.tensor(XYZ)
        A = torch.from_numpy(A) if isinstance(A, np.ndarray) else torch.tensor(A)
        W = torch.from_numpy(W) if isinstance(W, np.ndarray) else torch.tensor(W)
        return G, L, XYZ, A, W

    data = pd.read_csv(csv_file)
    try: cif_strings = data['cif']
    except: cif_strings = data['structure']

    p = multiprocessing.Pool(num_workers)
    partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max)
    results = p.map_async(partial_process_one, cif_strings).get()
    p.close()
    p.join()

    G, L, XYZ, A, W = zip(*results)

    # Convert to Tensors
    G = torch.tensor(np.array(G))
    A = torch.tensor(np.array(A)).reshape(-1, n_max)
    W = torch.tensor(np.array(W)).reshape(-1, n_max)
    XYZ = torch.tensor(np.array(XYZ)).reshape(-1, n_max, 3)
    L = torch.tensor(np.array(L)).reshape(-1, 6)

    # Sort
    A, XYZ = sort_atoms(W, A, XYZ)
    
    return G, L, XYZ, A, W

def GLXA_to_structure_single(G, L, X, A):
    """
    Convert G, L, X, A to pymatgen structure.
    """
    lattice = Lattice.from_parameters(*L)
    # filter out padding atoms
    idx = np.where(A > 0)
    A = A[idx]
    X = X[idx]
    structure = Structure.from_spacegroup(sg=G, lattice=lattice, species=A, coords=X).as_dict()

    return structure

def GLXA_to_csv(G, L, X, A, num_worker=1, filename='out_structure.csv'):

    # Ensure inputs are numpy for multiprocessing compatibility
    if isinstance(L, torch.Tensor): L = L.detach().cpu().numpy()
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(G, torch.Tensor): G = G.detach().cpu().numpy()
    if isinstance(G, int): G = np.array([G] * len(L))
    
    p = multiprocessing.Pool(num_worker)
    structures = p.starmap_async(GLXA_to_structure_single, zip(G, L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


if __name__=='__main__':
    atom_types = 119
    wyck_types = 28
    n_max = 24

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    # Example CSV path (dummy)
    # csv_file = 'test.csv'

    # To run this main block, ensure you have a valid csv or disable the GLXYZAW_from_file call
    # G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)
    
    # print (G.shape)
    # print (L.shape)
    # print (XYZ.shape)
    # print (A.shape)
    # print (W.shape)
    
    # print ('L:\n',L)
    # print ('XYZ:\n',XYZ)

    # Example of Lookup replacement
    def lookup_torch(G, W):
        # mult_table is (230, max_wyck)
        # G: (Batch,)
        # W: (Batch, n_max)
        table = torch.tensor(mult_table) # Move to tensor
        
        # Adjust G to 0-indexed
        G_idx = G - 1
        
        # We need to broadcast G to match W for gathering
        # G_idx: (Batch, 1) -> (Batch, n_max)
        G_expanded = G_idx.unsqueeze(1).expand(-1, W.shape[1])
        
        # Indexing: table[G, W]
        return table[G_expanded.long(), W.long()]

    # M = lookup_torch(G, W) 
    # print ('N:\n', M.sum(dim=-1))