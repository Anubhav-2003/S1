import torch
from torch import nn
from rope import RotaryPositionalEmbedding, computePosEmbd

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.nHeads = config.nHeads
        self.qLoraRank = config.qLoraRank
        self.kvLoraRank = config.kvLoraRank
        self.qkNopeRank = config.qkNopeRank
        self.qkRopeRank = config.qkRopeRank
        self.qkHeadRank = self.qkNopeRank + self.qkRopeRank
        self.vHeadRank = config.vHeadRank

        self.qD = nn.Linear(self.dim, self.qLoraRank)
        self.qDNorm = RMSNorm(self.qLoraRank)
        self.qU = nn.Linear(self.qLoraRank, self.nHeads * self.qkHeadRank)

        self.kvDandKRope = nn.Linear(self.dim, self.kvLoraRank + self.qkRopeRank)
        self.kvLoraNorm = RMSNorm(self.kvLoraRank)
        self.kvUp = nn.Linear(self.kvLoraRank, self.nHeads * (self.qkNopeRank + self.vHeadRank))
        self.wo = nn.Linear(self.nHeads * self.vHeadRank, self.dim)
        self.softmaxScale = self.qkHeadRank ** -0.5

        self.register_buffer("kvCache", torch.zeros(config.maxBatchSize, config.MaxContext, self.kvLoraRank), persistent=False)
        self.register_buffer("PosEmb", torch.zeros(config.maxBatchSize, config.MaxContext, self.qkRopeRank), persistent=False)

        self.rope = RotaryPositionalEmbedding(config.ropeConfig)
        
    def forward(self, X, start, freq):
        batchSize, seqLen, dim = X.size()
        endPos = start + seqLen
        
        q = self.qU(self.qDNorm(self.qD(X)))
        q = q.view(batchSize, seqLen, self.nHeads, self.qkHeadRank)
        qNope, qPos = torch.split(q, [self.qkNopeRank, self.qkRopeRank], dim = -1)
        kvloraAndKRope = self.kvDandKRope(X)
        kvloraAndKRope = kvloraAndKRope.view(batchSize, seqLen, self.nHeads, self.kvLoraRank + self.qkRopeRank)
        kvlora, kPos = torch.split(kvloraAndKRope, [self.kvLoraRank, self.qkRopeRank], dim = -1)
        cos = torch.cos(self.rope.base_freq[:seqLen])
        sin = torch.sin(self.rope.base_freq[:seqLen])

        qRot, kRot = computePosEmbd(qPos, kPos, cos, sin)
        


