import torch
from torch import nn
from rope import RotaryPositionalEmbedding, computePosEmbd
from rms import RMS as RMSNorm
from typing import Optional

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
        
    def forward(self, X, start, cos, sin, mask: Optional[torch.tensor]):
        batchSize, seqLen, dim = X.size()
        endPos = start + seqLen
        
        q = self.qU(self.qDNorm(self.qD(X)))
        q = q.view(batchSize, seqLen, self.nHeads, self.qkHeadRank) 
        qNope, qPos = torch.split(q, [self.qkNopeRank, self.qkRopeRank], dim = -1)
        kvloraAndKRope = self.kvDandKRope(X)
        kvlora, kPos = torch.split(kvloraAndKRope, [self.kvLoraRank, self.qkRopeRank], dim = -1)

        kPos = kPos.unsqueeze(2)
        qRot, kRot = computePosEmbd(qPos, kPos, cos, sin)

        with torch.no_grad():
            self.kvCache[:, start:endPos] = self.kvLoraNorm(kvlora)
            self.PosEmb[:, start:endPos] = kRot.squeeze(2)
        
        scoreR = torch.einsum("bshd,btd->bsht", qRot, self.PosEmb[:,:endPos])
        wUKandwUV = self.kvUp.weight
        wUKandwUV = wUKandwUV.view(self.nHeads, self.qkNopeRank + self.vHeadRank, self.kvLoraRank)
        wUK = wUKandwUV[:, :self.qkNopeRank, :]
        wUV = wUKandwUV[:, self.qkNopeRank: , :]
        QAbs = torch.einsum("bshd, hdc -> bshc", qNope, wUK)
        scoreC = torch.einsum("bshc, btc -> bsht", QAbs, self.kvCache[:, :endPos])
        combScore = (scoreR + scoreC) * self.softmaxScale
        if mask is not None:
            combScore += mask.unsqueeze(1)
        attnWeights = torch.softmax(combScore, dim = -1).to(X.dtype)

        outInter = torch.einsum("bsht, btc -> bshc", attnWeights, self.kvCache[:, :endPos])
        outHead = torch.einsum("bshc, hdc -> bshd", outInter, wUV)
        output = outHead.reshape(batchSize, seqLen, -1)
        output = self.wo(output)
        return output

