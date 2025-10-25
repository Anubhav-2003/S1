import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, config: S1Config):
        super().__init__()
        
        base_freq = self.computeBaseFreq(config)
        
        self.register_buffer("base_freq", base_freq, persistent=False)

    @staticmethod
    def computeBaseFreq(config: S1Config) -> torch.Tensor:
        base = config.base
        dim = config.dim
        device = config.device
        
        arangeTensor = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
        dimTensor = arangeTensor / dim
        
        baseFreq = 1.0 / (base ** dimTensor)
        
        return baseFreq