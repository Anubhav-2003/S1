import torch
import os
from torch import nn
from torch.optim import AdamW
from tokenizers import Tokenizer
import os
from transformer import Transformer
from dataPipe import DataPipe
from config import TrainConfig
from torch.amp import autocast, GradScaler
import glob

config = TrainConfig()

os.environ['HF_HOME'] = '/content/drive/MyDrive/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/content/drive/MyDrive/huggingface_cache/datasets'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

checkpointDir = "/content/drive/MyDrive/S1_Checkpoints"
maxSteps = 100000
checkpointFreq = 1000

device = torch.device(config.device)
print(f"Using device {device}.")

tokenizerPath = '../s1-tokenizer-32k.json'
if not os.path.exists(tokenizerPath):
    print(f"Error: Tokenizer file not found at {tokenizerPath}")
    print("Please run src/tokenizer.py first.")
    exit()

tokenizer = Tokenizer.from_file(tokenizerPath)
print("Tokenizer Loaded.")

savePath = "/content/drive/MyDrive/huggingface_cache/processed_dataset"

dataPipe = DataPipe(
    tokenizer=tokenizer,
    batchSize=config.batchSize,
    contextSize=config.maxContext,
    numOfThreads=config.numOfThreads,
    streaming=False,
    save_path=savePath
)

dataPipe.setup()
dataLoader = dataPipe.getDataLoader()

model = Transformer(config).to(device)
print(f"Model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

ce = nn.CrossEntropyLoss()

optimizer = AdamW(model.parameters(), lr=config.learning_rate)
scaler = GradScaler('cuda')

globalStep = 0
os.makedirs(checkpointDir, exist_ok=True)

checkpointList = sorted(glob.glob(f"{checkpointDir}/s1_checkpoint_*.pth"))
if checkpointList:
    latestCheckpointPath = checkpointList[-1]
    print(f"Loading checkpoint from {latestCheckpointPath}...")
    checkpoint = torch.load(latestCheckpointPath)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    globalStep = checkpoint['global_step']

    print(f"Successfully resumed training from Step {globalStep}")
else:
    print("No checkpoint found. Starting training from scratch at Step 0.")

model.train()
dataIterator = iter(dataLoader)

while globalStep < maxSteps:
    try:
        batch = next(dataIterator)
    except StopIteration:
        print(f"Completed one pass of the dataset. Restarting data iterator with shuffle.")
        dataIterator = iter(dataPipe.getDataLoader(shuffle=True))
        batch = next(dataIterator)

    tokenIds = batch['input_ids'].to(device)
    inputs = tokenIds[:, :-1]
    targets = tokenIds[:, 1:]

    optimizer.zero_grad(set_to_none=True)

    with autocast('cuda'):
        z = model(inputs, start=0)
        batchSize, seqLen, vocabSize = z.shape
        zFlat = z.reshape(batchSize * seqLen, vocabSize)
        targetsFlat = targets.reshape(batchSize * seqLen)
        loss = ce(zFlat, targetsFlat)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if globalStep % 10 == 0:
        print(f"Step {globalStep}/{maxSteps}, Loss: {loss.item():.4f}")

    if (globalStep % checkpointFreq == 0) and globalStep > 0:
        checkpointPath = f"{checkpointDir}/s1_checkpoint_{globalStep:08d}.pth"
        print(f"Saving checkpoint to {checkpointPath}...")
        torch.save({
            'global_step': globalStep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss.item(),
        }, checkpointPath)

        allCheckpoints = sorted(glob.glob(f"{checkpointDir}/s1_checkpoint_*.pth"))
        if len(allCheckpoints) > 3:
            os.remove(allCheckpoints[0])
            print(f"Removed old checkpoint: {allCheckpoints[0]}")

    globalStep += 1

print("Training finished.")
finalModelPath = f"{checkpointDir}/s1_model_final.pth"
torch.save(model.state_dict(), finalModelPath)
print(f"Model saved to {finalModelPath}")
