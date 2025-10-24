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
        return self.tokenizer(examples['text'], padding = False, truncation = False)
    
    def _createChunks(self, tokenizedExamples):
        stream = {k : sum(tokenizedExamples[k], []) for k in tokenizedExamples.keys()}
        totalLength = len(stream[list(tokenizedExamples.keys())[0]])

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

        tokenizdDataset = rawDataset.map(
            self._tokenize,
            batched = True,
            num_proc = self.numOfThreads,
            remove_columns = ['text']
        )

        self.finalDataset = tokenizdDataset.map(
            self._createChunks,
            batched = True,
            num_proc = self.numOfThreads
        )

        self.finalDataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask'])

    def getDataLoader(self, shuffle = True):
        if self.finalDataset is None:
            raise RuntimeError("Pipeline has not been set up. Please call .setup() first.")
        
        return DataLoader(
            self.finalDataset,
            batch_size = self.batchSize,
            shuffle = shuffle,
            num_workers = self.numOfThreads,
            pin_memory = True
        )
