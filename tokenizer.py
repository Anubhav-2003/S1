from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

books = load_dataset("rojagtap/bookcorpus", split = 'train', streaming = True)

C4Full = load_dataset("allenai/c4", 
    "en", 
    split='train',
    streaming=True)

C4 = C4Full.take(2500000)
fullDataSet = interleave_datasets([books, C4])

def getTrainingCorpus(batchSize = 1000):
    iterator = iter(fullDataSet)
    while True:
        try:
            batch = [next(iterator)['text'] for _ in range(batchSize)]
            batch = [text for text in batch if text]

            if not batch:
                continue

            yield batch

        except StopIteration:
            print("Processing of Dataset Done.")
            break
        except Exception as e:
            print(f"Skipping a bad batch due to error: {e}")
            continue

S1_TOKENIZER = Tokenizer(BPE(unk_token="[UNK]"))
S1_TOKENIZER.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=32000,
    special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
)

S1_TOKENIZER.train_from_iterator(
    getTrainingCorpus(),
    trainer = trainer
)


out = "s1-tokenizer-32k.json"
S1_TOKENIZER.save(out)


