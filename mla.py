import torch
from torch import nn

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


