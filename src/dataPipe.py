from datasets import load_dataset, interleave_datasets, load_from_disk
from torch.utils.data import DataLoader
import os
class DataPipe:
    def __init__(self, tokenizer, batchSize, contextSize, numOfThreads, streaming, save_path = None):
        self.tokenizer = tokenizer
        self.batchSize = batchSize
        self.contextSize = contextSize
        self.numOfThreads = numOfThreads
        self.streaming = streaming
        self.finalDataset = None
        self.save_path = save_path
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
        if self.save_path and os.path.exists(self.save_path):
            print(f"Loading pre-processed dataset from {self.save_path}...")
            self.finalDataset = load_from_disk(self.save_path)
            self.finalDataset = self.finalDataset.with_format(type='torch')
            print("Dataset setup complete. Ready for DataLoader.")
            return
        
        book_id = "rojagtap/bookcorpus"
        wiki_id = "wikimedia/wikipedia"
        
        wiki_config = "20231101.en" 

        print(f"Loading datasets '{book_id}' and '{wiki_id}' (config: '{wiki_config}')...")
        print("This will use your local cache in GDrive automatically.")
        
        try:
            books = load_dataset(book_id, split = 'train', streaming=self.streaming)
            wiki = load_dataset(wiki_id, wiki_config, split='train', streaming=self.streaming)
        
        except Exception as e:
            print(f"--- ERROR ---")
            print(f"Failed to load datasets. This could be because the config '{wiki_config}' for Wikipedia is wrong.")
            print("Please check your cache folder '/content/drive/MyDrive/huggingface_cache/datasets/wikimedia__wikipedia' to see what the config subfolder is named (e.g., '20231101.en', 'en', etc.)")
            print(f"Original error: {e}")
            raise

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
