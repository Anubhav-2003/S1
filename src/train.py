import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch import nn
from torch.optim import AdamW
from tokenizers import Tokenizer
import os
from transformer import Transformer
from dataPipe import DataPipe
from config import TrainConfig
from torch.amp import autocast, GradScaler
config = TrainConfig()

device = torch.device(config.device)
print(f"Using device {device}.")

tokenizerPath = '../s1-tokenizer-32k.json'
if not os.path.exists(tokenizerPath):
    print(f"Error: Tokenizer file not found at {tokenizerPath}")
    print("Please run src/tokenizer.py first.")
    exit()

tokenizer = Tokenizer.from_file(tokenizerPath)
print("Tokenizer Loaded.")

dataPipe = DataPipe(
    tokenizer=tokenizer,
    batchSize=config.batchSize,
    contextSize=config.maxContext,
    numOfThreads=config.numOfThreads,
    streaming=True
)

dataPipe.setup()
dataLoader = dataPipe.getDataLoader()

model = Transformer(config).to(device)
print(f"Model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

CE = nn.CrossEntropyLoss()

optimizer = AdamW(model.parameters(), lr = config.learning_rate)
scaler = GradScaler('cuda')

for epoch in range(config.epochs):
    model.train()
    for i, batch in enumerate(dataLoader):
        tokenIds = batch['input_ids'].to(device)
        inputs = tokenIds[:, :-1]
        targets = tokenIds[:, 1:]

        optimizer.zero_grad(set_to_none=True) 
        
        with autocast('cuda'):
            Z = model(inputs, start = 0)
            batchSize, seqLen, vocabSize = Z.shape

            ZFlat = Z.reshape(batchSize * seqLen, vocabSize)
            targetsFlat = targets.reshape(batchSize * seqLen)

            loss = CE(ZFlat, targetsFlat)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        print(f"Epoch {epoch+1}/{config.epochs}, Step {i}, Loss: {loss.item():.4f}")

print("Training finished.")

torch.save(model.state_dict(), 'transformer_model.pth')
print("Model saved to transformer_model.pth")
