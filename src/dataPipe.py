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
        valid_texts = [t for t in examples['text'] if t is not None and len(t.strip()) > 0]
        
        if not valid_texts:
            return {
                "input_ids": [],
                "attention_mask": []
            }
            
        encoded_batch = self.tokenizer.encode_batch(valid_texts)
        
        return {
            "input_ids": [encoding.ids for encoding in encoded_batch],
            "attention_mask": [encoding.attention_mask for encoding in encoded_batch]
        }
    
    def _createChunks(self, tokenizedExamples):
        keys_to_chunk = ['input_ids', 'attention_mask']
        
        stream = {}
        for k in keys_to_chunk:
            valid_chunks = [chunk for chunk in tokenizedExamples[k] if chunk is not None and chunk]
            stream[k] = sum(valid_chunks, [])
        
        totalLength = len(stream.get('input_ids', []))

        if totalLength == 0:
            return {k: [] for k in keys_to_chunk}
            
        totalLength = (totalLength // self.contextSize) * self.contextSize

        if totalLength == 0:
            return {k: [] for k in keys_to_chunk}

        chunkedExamples = {
            k: [t[i: i + self.contextSize] for i in range(0, totalLength, self.contextSize)]
            for k, t in stream.items()
        }
        
        return chunkedExamples
    

    def setup(self):
        book_path = "/content/drive/MyDrive/huggingface_cache/datasets/rojagtap__bookcorpus" 
        wiki_path = "/content/drive/MyDrive/huggingface_cache/datasets/wikimedia__wikipedia"

        print(f"Loading datasets from {book_path} and {wiki_path} (non-streaming)...")
        
        books = load_dataset(book_path, split = 'train')
        wiki = load_dataset(wiki_path, split='train')
        
        rawDataset = interleave_datasets([books, wiki])
        print("Datasets loaded and interleaved.")

        filteredDataset = rawDataset.filter(
            lambda example: example['text'] is not None and len(example['text'].strip()) > 0,
            num_proc = self.numOfThreads
        )
        print("Filtering done.")

        tokenizdDataset = filteredDataset.map(
            self._tokenize,
            batched = True,
            remove_columns = rawDataset.column_names,
            num_proc = self.numOfThreads
        )
        print("Tokenization done.")

        chunkedDataset = tokenizdDataset.map(
            self._createChunks,
            batched = True,
            remove_columns = tokenizdDataset.column_names,
            num_proc = self.numOfThreads
        )
        print("Chunking done.")

        self.finalDataset = chunkedDataset.filter(
            lambda example: example['input_ids'] is not None and len(example['input_ids']) > 0,
            num_proc = self.numOfThreads
        )
        print("Final filtering done.")
        
        self.finalDataset = self.finalDataset.with_format(type = 'torch')
        print("Dataset setup complete. Ready for DataLoader.")

    def getDataLoader(self, shuffle = True):
        if self.finalDataset is None:
            raise RuntimeError("Pipeline has not been set up. Please call .setup() first.")
        
        print(f"Creating DataLoader with shuffle={shuffle} and num_workers={self.numOfThreads}")
        
        return DataLoader(
            self.finalDataset,
            batch_size = self.batchSize,
            shuffle = shuffle, 
            num_workers = self.numOfThreads, 
            pin_memory = True
        )