"""
Data loading and preprocessing for the Reverse Dictionary model.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from config import DEVICE


class ReverseDictDataset(Dataset):
    """
    PyTorch Dataset for English-German reverse dictionary training.
    
    Matches definitions with their target GloVe word vectors.
    """
    
    def __init__(self, csv_file, tokenizer, glove_vectors, max_len=64):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV with 'Definition' and 'Word' columns
            tokenizer: BERT tokenizer for encoding definitions
            glove_vectors: GloVe word vectors (from gensim)
            max_len: Maximum token length for definitions (default: 64)
        """
        df = pd.read_csv(csv_file).dropna(subset=['Definition', 'Word'])
        self.data = []
        
        print("Preprocessing dataset and matching GloVe vectors...")
        
        for _, row in df.iterrows():
            word = str(row['Word']).lower().strip()
            
            # Only include words that exist in GloVe vectors
            if word in glove_vectors:
                self.data.append({
                    'word': word,
                    'definition': str(row['Definition']),
                    'vector': torch.tensor(glove_vectors[word], dtype=torch.float32)
                })
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Dataset Ready: {len(self.data)} valid pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            Dictionary with:
            - input_ids: Tokenized definition
            - attention_mask: Padding mask
            - targets: Target 300-dim GloVe vector
            - word: The target word
        """
        item = self.data[idx]
        
        # Tokenize definition using BERT tokenizer
        encoding = self.tokenizer(
            item['definition'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'targets': item['vector'],
            'word': item['word']
        }


def load_tokenizer_and_vectors(model_name, vector_source='gensim'):
    """
    Load BERT tokenizer and word vectors.
    
    Args:
        model_name: Name of BERT model
        vector_source: 'gensim' to download via gensim
        
    Returns:
        Tuple of (tokenizer, vectors)
    """
    from transformers import BertTokenizer
    
    if vector_source == 'gensim':
        import gensim.downloader as api
        from config import GLOVE_MODEL
        
        print(f"Loading {GLOVE_MODEL}...")
        vectors = api.load(GLOVE_MODEL)
    
    print(f"Loading {model_name} tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    return tokenizer, vectors
