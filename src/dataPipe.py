from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader

class DataPipe:
    def __init__(self, tokenizer, batchSize, contextSize, numOfThreads, streaming):
        self.tokenizer = tokenizer
        self.batchSize = batchSize
        self.contextSize = contextSize
        self.numOfThreads = numOfThreads
        self.streaming = streaming
        self.finalDataset = None
        return
    
    def _tokenize(self, examples):
        encoded_batch = self.tokenizer.encode_batch(examples['text'])
        
        return {
            "input_ids": [encoding.ids for encoding in encoded_batch],
            "attention_mask": [encoding.attention_mask for encoding in encoded_batch]
        }
    
    def _createChunks(self, tokenizedExamples):
        keys_to_chunk = ['input_ids', 'attention_mask']
        
        stream = {}
        for k in keys_to_chunk:
            valid_chunks = [chunk for chunk in tokenizedExamples[k] if chunk is not None]
            stream[k] = sum(valid_chunks, [])
        
        totalLength = len(stream['input_ids'])
        totalLength = (totalLength // self.contextSize) * self.contextSize

        chunkedExamples = {
            k: [t[i: i + self.contextSize] for i in range(0, totalLength, self.contextSize)]
            for k, t in stream.items()
        }
        
        return chunkedExamples
    

    def setup(self):
        books = load_dataset("rojagtap/bookcorpus", split = 'train', streaming = True)
        C4Full = load_dataset("allenai/c4", 
            "en", 
            split='train',
            streaming=True)

        C4 = C4Full.take(2500000) 
        rawDataset = interleave_datasets([books, C4])

        filteredDataset = rawDataset.filter(
            lambda example: example['text'] is not None and len(example['text'].strip()) > 0
        )
        tokenizdDataset = filteredDataset.map(
            self._tokenize,
            batched = True,
        )

        self.finalDataset = tokenizdDataset.map(
            self._createChunks,
            batched = True
        )

        self.finalDataset = self.finalDataset.with_format(type = 'torch')

    def getDataLoader(self, shuffle = True):
        if self.finalDataset is None:
            raise RuntimeError("Pipeline has not been set up. Please call .setup() first.")
        
        return DataLoader(
            self.finalDataset,
            batch_size = self.batchSize,
            shuffle = False,
            num_workers = self.numOfThreads,
            pin_memory = True
        )
