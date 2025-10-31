import torch
from torch import nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocabSize, dim):
        super().__init__()
        self.vocabSize = vocabSize
        self.dim = dim
        self.W = nn.Parameter(torch.empty(self.vocabSize, self.dim))

    def forward(self, X):
        return F.embedding(X, self.W)