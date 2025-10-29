import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        base_freq = self.computeBaseFreq(config)
        
        self.register_buffer("base_freq", base_freq, persistent=False)

    @staticmethod
    def computeBaseFreq(config) -> torch.Tensor:
        base = config.base
        dim = config.dim
        device = config.device
        
        arangeTensor = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        dimTensor = arangeTensor / dim
        
        baseFreq = 1.0 / (base ** dimTensor)
        
        return baseFreq
    
def computeOrtogonalPairs(X):
    XFirst = X[..., :X.shape[-1] // 2]
    XSecond = X[..., X.shape[-1] // 2 : ]
    return torch.cat((-XSecond, XFirst), dim=-1)
    
def computePosEmbd(Q, K, cos, sin, unsqueezeDim = 1):
    cos = cos.unsqueeze(unsqueezeDim)
    sin = sin.unsqueeze(unsqueezeDim)
    Q_rot = Q * cos + computeOrtogonalPairs(Q) * sin
    K_rot = K * cos + computeOrtogonalPairs(K) * sin
    return Q_rot, K_rot

