import torch
from torch import nn
from mla import MultiHeadLatentAttention
from mlp import MLP
from rms import RMS as RMSNorm
from embedding import Embedding
from rope import RotaryPositionalEmbedding
from typing import Optional

class TransformerBlock(nn.Module):
    def __init__(self, layerID, config):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config.MLACONFIG)
        self.mlp = MLP(config.MLP)
        self.attnNorm = RMSNorm(config.dim)
        self.mlpNorm = RMSNorm(config.dim)

    def forward(self, X, start, cos, sin, mask: Optional[torch.tensor]):
        X = X + self.attention(self.attnNorm(X), start, cos, sin, mask)
        X = X + self.mlp(self.mlpNorm(X)) 
        return X
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.maxContext = config.maxContext
        self.Emb = Embedding(config.vocabSize, config.dim)
        self.layers = nn.ModuleList()
        for i in range(config.nLayers):
            self.layers.append(TransformerBlock(i, config))
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocabSize)
        self.rope = RotaryPositionalEmbedding(config)
        baseFreq = self.rope.base_freq
        m = torch.arange(self.maxContext, device=config.device, dtype=torch.float32)
        thetas = torch.outer(m, baseFreq)
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, input, start = 0):
        batchSize, seqLen = input.size()

        h = self.Emb(input)
        cos = self.cos_cache[start: start + seqLen]
        sin = self.sin_cache[start: start + seqLen]

        mask = None
        if seqLen > 1:
            mask = torch.full((seqLen, seqLen), float("-inf"), device=input.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start, cos, sin, mask)
        h = self.norm(h)
        Z = self.head(h)
        return Z