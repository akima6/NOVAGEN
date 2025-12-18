import torch

class CrystalFormerGenerator:
    def __init__(self, model: torch.nn.Module, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def generate_logits(self, G, XYZ, A, W, M):
        G = torch.tensor(G, device=self.device)
        XYZ = torch.tensor(XYZ, device=self.device)
        A = torch.tensor(A, device=self.device)
        W = torch.tensor(W, device=self.device)
        M = torch.tensor(M, device=self.device)

        with torch.no_grad():
            out = self.model(G, XYZ, A, W, M, is_train=False)
        return out
