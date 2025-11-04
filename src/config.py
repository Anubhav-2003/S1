import torch

try:
    import torch_xla.core.xla_model as xm
    TPU_DEVICE = xm.xla_device()
    print("Successfully found TPU device!")
except ImportError:
    TPU_DEVICE = None
    print("torch_xla not found, TPU not available.")

class MLACONFIG:
    dim = 768
    nHeads = 6
    qLoraRank = 512
    kvLoraRank = 256
    qkNopeRank = 768
    qkRopeRank = 64
    vHeadRank = 768
    maxBatchSize = 8
    MaxContext = 1024

class MLPCONFIG:
    dim = 768
    hiddenDim = 768

class TrainConfig:
    dim = 768 
    nLayers = 7
    vocabSize = 32000
    maxContext = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = 10000
    MLACONFIG = MLACONFIG
    MLP = MLPCONFIG

    batchSize = 8
    numOfThreads = 4
    learning_rate = 3e-4
    epochs = 1