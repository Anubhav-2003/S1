import torch
from torch import nn
import torch.nn.functional as F

class RMS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-6
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, X):
        return F.rms_norm(X, (self.dim, ), self.w, self.epsilon)