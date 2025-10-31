import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.hiddenDim = config.hiddenDim
        self.w1 = nn.Linear(self.dim, self.hiddenDim)
        self.w3 = nn.Linear(self.dim, self.hiddenDim)
        self.w2 = nn.Linear(self.hiddenDim, self.dim)

    def forward(self, X: torch.tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(X)) * self.w3(X))