import torch
from torch import nn
from mla import MultiHeadLatentAttention
from mlp import MLP
from rms import RMS as RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, layerID, config):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config.MLACONFIG)
        self.mlp = MLP(config.MLP)
        self.attnNorm = RMSNorm(config.dim)
        self.mlpNorm = RMSNorm(config.dim)

    def forward(self, X, start, baseFreq):
        X = X + self.attention(self.attnNorm(X), start, baseFreq)
        X = X + self.mlp(self.mlpNorm(X)) 
        return X
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.maxContext = config.maxContext