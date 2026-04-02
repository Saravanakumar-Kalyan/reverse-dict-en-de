# Cross-Lingual Reverse Dictionary (English-German)

A neural network-based reverse dictionary system that finds words based on their definitions. The model uses a hybrid BERT-LSTM architecture and supports both English and cross-lingual (English-to-German) lookups.

## Project Overview

### Key Features

- **Hybrid Architecture**: Combines BERT (for contextual understanding) with Bi-LSTM (for sequence dependencies)
- **Two-Stage Training**: 
  - Epochs 1-4: BERT frozen, only LSTM/Linear layers train
  - Epochs 5+: BERT fine-tuned with lower learning rate
- **Cross-Lingual Support**: Maps English word vectors to German space using Orthogonal Procrustes
- **Word Alignment**: Uses FastAlign for computing English-German word correspondences
- **Vector Space**: Leverages pre-trained GloVe and FastText embeddings

## Repository Structure

```
reverse-dict-en-de/
├── config.py              # Configuration and hyperparameters
├── model.py               # BERT-LSTM model architecture
├── data_processing.py     # Dataset loading and tokenization
├── training.py            # Training loop and utilities
├── word_alignment.py      # Word alignment and corpus processing
├── vector_mapping.py      # Cross-lingual vector mapping (Procrustes)
├── inference.py           # Inference for dictionary lookup
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore file
└── notebooks/             # Original Jupyter notebooks
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Saravanakumar-Kalyan/reverse-dict-en-de.git
   cd nlp-reverse-dict
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. For cross-lingual features, install FastAlign dependencies:
   ```bash
   sudo apt-get install libgoogle-perftools-dev libsparsehash-dev cmake
   git clone https://github.com/clab/fast_align.git
   cd fast_align
   mkdir build && cd build
   cmake .. && make
   ```

## Usage

### Training

To train a new model on the English dictionary:

```bash
python main.py --train
```

Options:
- `--no-plot`: Skip plotting training history
- `--model PATH`: Specify custom model save path

**Required Data File**: `OPTED-Dictionary.csv` with columns:
- `Word`: Target English word
- `Definition`: Definition of the word

### Inference Only

Run inference with a pre-trained model:

```bash
python main.py --inference --model best_reverse_dict_model.pth
```

### Complete Pipeline

To run the full pipeline including word alignment and cross-lingual mapping:

```bash
python main.py --train
```

**Required Files**:
- `OPTED-Dictionary.csv`: English dictionary
- `epuds.en-de.en`: Parallel English sentences (Europarl dataset)
- `epuds.en-de.de`: Parallel German sentences
- `fast_align_en_de`: FastAlign output (computed from parallel corpus)

## Module Documentation

### config.py
Central configuration file with all hyperparameters:
- Device (CPU/GPU)
- Model paths and names
- Training hyperparameters
- Data file paths

### model.py
**BertLSTMReverseDict**: Main model class
- BERT encoder (768-dim)
- Bi-LSTM decoder (256 units)
- Linear projection (300-dim for GloVe)

Key functions:
- `load_model_for_inference()`: Load checkpoint for inference
- `freeze_bert()` / `unfreeze_bert()`: Control BERT parameters

### data_processing.py
**ReverseDictDataset**: PyTorch Dataset class
- Loads definitions and their target word vectors
- Tokenizes using BERT tokenizer
- Matches with GloVe vectors

### training.py
**run_training()**: Complete training pipeline
- Handles data loading and splitting
- Implements two-stage training strategy
- Saves best model checkpoint

### word_alignment.py
Word alignment utilities:
- `tokenize()`: Separate punctuation from words
- `prepare_fast_align_data()`: Format parallel corpus for FastAlign
- `clean_parallel_corpus()`: Remove empty sentence pairs
- `load_alignments()`: Parse FastAlign output
- `collapse_dictionary()`: Simplify alignment results

### vector_mapping.py
**learn_vector_mapping()**: Orthogonal Procrustes algorithm
- Learns transformation matrix W to map English to German vector space
- Preserves geometric properties (distances, angles)
- Returns 300×300 orthogonal matrix

### inference.py
Two inference functions:

1. **reverse_lookup_english()**
   - Input: English definition
   - Output: Top 5 English words

2. **reverse_lookup_german()**
   - Input: English definition
   - Output: Top 5 German words
   - Uses transformation matrix W

## Training Details

### Architecture

```
Definition (text)
    ↓
BERT Tokenizer
    ↓
BERT Encoder (768-dim)
    ↓
Bi-LSTM Processing (512-dim)
    ↓
Dropout
    ↓
Linear Layer (300-dim)
    ↓
Predicted Word Vector
```

### Loss Function

Mean Squared Error (MSE) between predicted and target GloVe vectors.

### Two-Stage Training

**Phase 1 (Epochs 1-4)**: BERT Frozen
- Only LSTM and Linear layers learn
- Prevents destroying pre-trained BERT weights
- Learning rate: 1e-3

**Phase 2 (Epochs 5-15)**: BERT Fine-Tuning
- All layers can learn
- Lower learning rate: 1e-5 (subtle adaptations)
- Dynamic learning rate reduction on plateau

## Cross-Lingual Mapping

### Orthogonal Procrustes Algorithm

Transforms English vectors to German space:

1. **Collect aligned pairs**: English-German word translations from FastAlign
2. **Compute W**: Minimize ||XW - Y||_F where X=English vectors, Y=German vectors
3. **Constraint**: W must be orthogonal (distance-preserving)

### Result

Single transformation matrix W (300×300) that can transform any English word vector to German space.

## Example Usage

```python
from model import load_model_for_inference
from inference import reverse_lookup_english, reverse_lookup_german
import gensim.downloader as api

# Load model and vectors
model, tokenizer = load_model_for_inference("best_reverse_dict_model.pth")
glove_vectors = api.load("glove-wiki-gigaword-300")

# English lookup
definition = "A large fruit with a hard green skin"
results = reverse_lookup_english(model, definition, tokenizer, glove_vectors)
print("English predictions:", results)

# Cross-lingual lookup
W = np.load("transformation_matrix.npy")
de_vectors = api.load("fasttext-wiki-news-subwords-300")
results = reverse_lookup_german(model, definition, tokenizer, W, de_vectors)
print("German predictions:", results)
```

## Performance

- **Training Time**: ~2 hours (15 epochs, 1 GPU)
- **Inference Speed**: ~0.1-0.5 sec per definition (GPU)
- **Dictionary Size**: ~10,000+ valid words (after GloVe filtering)

## Datasets

### OPTED Dictionary
English words with their definitions
- Source: The Online Plain Text English Dictionary
- Processing: Filter words present in GloVe vocabulary

### Europarl Parallel Corpus
English-German parallel sentences
- Source: European Parliament proceedings
- Processing: Tokenize and format for FastAlign

### Word Vectors
- **English (GloVe)**: glove-wiki-gigaword-300 (300-dim, 2.2M words)
- **German (FastText)**: fasttext-wiki-news-subwords-300 (300-dim, multilingual)

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| transformers | BERT models and tokenizers |
| gensim | Word vector loading/management |
| pandas | Data processing |
| numpy | Numerical computing |
| scipy | Orthogonal Procrustes |
| matplotlib | Visualization |


## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Use GPU with more memory
- Run on CPU (slower but works)

### Low Accuracy
- Ensure dictionary has sufficient word-definition pairs
- Check that GloVe vectors are properly loaded
- Verify data preprocessing (tokenization)
- Train longer (increase EPOCHS)

### FastAlign Issues
- Install required C++ libraries: `libgoogle-perftools-dev`, `libsparsehash-dev`
- Use Linux/Mac (Windows support requires WSL or Docker)
- Verify parallel corpus is properly formatted

