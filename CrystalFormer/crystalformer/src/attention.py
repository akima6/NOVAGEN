import torch
import torch.nn as nn

class MultiHeadAttentionTorch(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        model_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size

        self.q_proj = nn.Linear(model_size, model_size)
        self.k_proj = nn.Linear(model_size, model_size)
        self.v_proj = nn.Linear(model_size, model_size)
        self.out_proj = nn.Linear(model_size, model_size)

    def forward(self, q, k, v, mask=None):
        # TEMPORARY STUB: no real attention yet
        # This is only to unblock the JAX → PyTorch migration
        return self.out_proj(self.v_proj(v))
