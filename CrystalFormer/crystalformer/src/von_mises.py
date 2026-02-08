import torch
import torch.distributions as dist
import numpy as np

def sample_von_mises(key, loc, concentration, shape):
    """
    Generate sample from von Mises distribution.
    
    Args:
        key: Ignored in PyTorch (uses global RNG)
        loc: (Batch, ...) or scalar
        concentration: (Batch, ...) or scalar
        shape: Output shape. 
               Note: if loc/conc are broadcastable to shape, we sample naturally.
    
    Returns:
        samples: Tensor of shape 'shape'
    """
    # Ensure inputs are tensors
    if not isinstance(loc, torch.Tensor):
        loc = torch.tensor(loc)
    if not isinstance(concentration, torch.Tensor):
        concentration = torch.tensor(concentration)
        
    device = loc.device
    
    # concentration must be > 0. Handle 0 case (uniform) separately?
    # PyTorch VonMises requires concentration > 0.
    # If concentration is 0, it's Uniform(-pi, pi).
    # We replace 0 with a tiny epsilon for compatibility, or handle explicitly.
    # The original JAX code handles it naturally via the rejection sampler math.
    # Here, let's clamp min concentration.
    concentration = torch.clamp(concentration, min=1e-6)
    
    # Create distribution
    # PyTorch VonMises expects loc and concentration.
    # Broadcast to shape if needed?
    # Usually shape comes from input dimensions.
    # If shape is provided explicitly, we expand inputs.
    
    # If inputs are scalars, expand to shape
    if loc.numel() == 1 and concentration.numel() == 1:
         loc = loc.expand(shape)
         concentration = concentration.expand(shape)
    
    vm = dist.von_mises.VonMises(loc, concentration)
    
    # Sample
    # VonMises.sample(sample_shape) adds dims.
    # If loc/conc already match 'shape', we pass sample_shape=torch.Size([]).
    # If loc/conc are (B,), and shape is (B,), then sample_shape=[]
    # If shape argument implies *additional* dims, we pass them.
    # Original usage: sample_von_mises(..., (batchsize,)) where inputs are (batchsize,).
    # So we want 1 sample per param.
    
    samples = vm.sample()
    
    # Wrap to [-pi, pi] (PyTorch VM is usually centered? No, it's on circle).
    # PyTorch VonMises support is usually [-pi, pi].
    # Original JAX code: (samples + pi) % 2pi - pi.
    # Let's ensure range.
    samples = (samples + np.pi) % (2.0 * np.pi) - np.pi
    
    return samples

def von_mises_logpdf(x, loc, concentration):
    '''
    kappa is the concentration. kappa = 0 means uniform distribution.
    Computes log(exp(k*cos(x-loc)) / (2*pi*I0(k)))
    Using i0e stability trick: log(I0(k)) = k + log(I0e(k))
    '''
    # Ensure tensors
    if not isinstance(x, torch.Tensor): x = torch.tensor(x)
    if not isinstance(loc, torch.Tensor): loc = torch.tensor(loc)
    if not isinstance(concentration, torch.Tensor): concentration = torch.tensor(concentration)
    
    concentration = torch.clamp(concentration, min=1e-6)
    
    # Term 1: - log(2pi)
    term1 = -np.log(2 * np.pi)
    
    # Term 2: - log(I0(k)) = - (k + log(I0e(k)))
    # PyTorch: torch.special.i0e
    term2 = -(concentration + torch.log(torch.special.i0e(concentration)))
    
    # Term 3: k * cos(x - loc)
    # Original JAX: concentration * (jnp.cos((x - loc) % (2 * jnp.pi)) - 1)
    # Wait, original JAX implementation:
    # -(log(2pi) + log(i0e(k))) + k*(cos(x-loc)-1)
    # Let's expand log(i0e(k)):
    # log(I0(k) * exp(-k)) = log(I0(k)) - k
    # So -(... + log(I0(k)) - k) = -log(...) - log(I0(k)) + k
    # Plus k*(cos - 1) = k*cos - k
    # Result: -log(2pi) - log(I0(k)) + k - k + k*cos - k ?
    # Let's re-verify JAX formula:
    # return -(log(2pi) + log(i0e(k))) + k*(cos - 1)
    # = -log(2pi) - (log(I0(k)) - k) + k*cos - k
    # = -log(2pi) - log(I0(k)) + k + k*cos - k
    # = -log(2pi) - log(I0(k)) + k*cos
    # This matches the standard definition.
    
    # Note on modulo: (x - loc) % 2pi ensures we compute shortest distance on circle?
    # Cos is periodic, so % 2pi is redundant but safe.
    
    return term1 + term2 + concentration * (torch.cos((x - loc)) - 1.0)


if __name__=='__main__':
    # Test
    loc = torch.tensor([-1.0, 1.0, 0.0])
    kappa = torch.tensor([10.0, 10.0, 100.0])
    
    # Shape inferred from inputs
    x = sample_von_mises(None, loc, kappa, (3,))
    print(x)
    
    logp = von_mises_logpdf(x, loc, kappa)
    print(logp)