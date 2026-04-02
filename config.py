"""
Configuration settings for the Cross-Lingual Reverse Dictionary project.
"""

import torch

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL CONSTANTS ---
GLOVE_MODEL = "glove-wiki-gigaword-300"
FASTTEXT_MODEL = "fasttext-wiki-news-subwords-300"
BERT_MODEL = 'bert-base-uncased'
SAVE_PATH = "best_reverse_dict_model.pth"

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 32
MAX_LEN = 64
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
EPOCHS = 15
BERT_FREEZE_UNTIL_EPOCH = 4  # Unfreeze BERT at epoch 4
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.4

# --- SCHEDULER ---
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2

# --- DATA PATHS ---
DICT_CSV_FILE = "OPTED-Dictionary.csv"
SOURCE_FILE = "epuds.en-de.en"
TARGET_FILE = "epuds.en-de.de"
ALIGNMENT_FILE = "fast_align_en_de"
CLEANED_SOURCE = "cleaned.en"
CLEANED_TARGET = "cleaned.de"

# --- OUTPUT PATHS ---
FORWARD_ALIGN = "forward.align"
REVERSE_ALIGN = "reverse.align"
PROCESSED_DATA = "text.en-de"

# --- INFERENCE ---
TOP_K_PREDICTIONS = 5
