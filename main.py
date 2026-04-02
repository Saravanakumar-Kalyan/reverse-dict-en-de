"""
Main execution script for training the Reverse Dictionary model.

This script orchestrates the complete pipeline:
1. Load data and train English reverse dictionary model
2. Process word alignments
3. Learn cross-lingual vector mapping
4. Perform inference in both English and German
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import gensim.downloader as api

from config import (
    DEVICE, DICT_CSV_FILE, SOURCE_FILE, TARGET_FILE, ALIGNMENT_FILE,
    CLEANED_SOURCE, CLEANED_TARGET, PROCESSED_DATA, GLOVE_MODEL, 
    FASTTEXT_MODEL, BERT_MODEL
)
from training import run_training, plot_training_history
from word_alignment import load_alignments, collapse_dictionary
from vector_mapping import learn_vector_mapping
from inference import load_for_inference, reverse_lookup_english, reverse_lookup_german


def setup_environment():
    """Verify that required data files exist."""
    required_files = [DICT_CSV_FILE, SOURCE_FILE, TARGET_FILE, ALIGNMENT_FILE]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("Warning: Some data files are missing:")
        for f in missing:
            print(f"  - {f}")
        print("\nYou may need to download the required datasets first.")
    
    return all(not missing)


def train_model(csv_file=DICT_CSV_FILE, plot=True):
    """Train the BERT-LSTM reverse dictionary model."""
    
    print("\n" + "="*70)
    print("  STAGE 1: Training English Reverse Dictionary Model")
    print("="*70)
    
    if not os.path.exists(csv_file):
        print(f"Error: Dictionary file '{csv_file}' not found.")
        print("Please provide a CSV file with 'Word' and 'Definition' columns.")
        return None, None, None, None
    
    # Train model
    model, history, tokenizer, glove_vectors = run_training(csv_file)
    
    # Plot training history
    if plot:
        plot_training_history(history, save_path="training_history.png")
    
    return model, history, tokenizer, glove_vectors


def setup_word_alignment(alignment_file=ALIGNMENT_FILE, cleaned_src=CLEANED_SOURCE, 
                        cleaned_tgt=CLEANED_TARGET):
    """Load and process word alignments."""
    
    print("\n" + "="*70)
    print("  STAGE 2: Word Alignment and Dictionary Construction")
    print("="*70)
    
    if not os.path.exists(alignment_file):
        print(f"Error: Alignment file '{alignment_file}' not found.")
        print("Please run FastAlign first to generate alignments.")
        return None
    
    # Load alignments
    print(f"\nLoading alignments from {alignment_file}...")
    alignment_dict = load_alignments(alignment_file, cleaned_src, cleaned_tgt)
    
    # Collapse and clean dictionary
    print("\nCleaning dictionary...")
    collapsed_dict = collapse_dictionary(alignment_dict)
    
    return collapsed_dict


def setup_cross_lingual_mapping(collapsed_dict):
    """Learn transformation matrix for cross-lingual vector mapping."""
    
    print("\n" + "="*70)
    print("  STAGE 3: Cross-Lingual Vector Space Mapping")
    print("="*70)
    
    if not collapsed_dict:
        print("Error: No collapsed dictionary provided.")
        return None, None, None
    
    # Load vectors
    print(f"\nLoading {GLOVE_MODEL}...")
    glove_vectors = api.load(GLOVE_MODEL)
    
    print(f"Loading {FASTTEXT_MODEL}...")
    fasttext_vectors = api.load(FASTTEXT_MODEL)
    
    # Learn transformation matrix using Orthogonal Procrustes
    print("\nLearning cross-lingual transformation matrix...")
    W = learn_vector_mapping(collapsed_dict, glove_vectors, fasttext_vectors)
    
    return W, glove_vectors, fasttext_vectors


def demo_inference(W, glove_vectors, fasttext_vectors, model, tokenizer):
    """Run demo inference."""
    
    print("\n" + "="*70)
    print("  STAGE 4: Inference Demo")
    print("="*70)
    
    test_queries = [
        "A large fruit with a hard green skin and red juicy flesh",
        "A feeling of great pleasure or happiness",
        "A person who studies science"
    ]
    
    print("\n--- English Reverse Dictionary ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = reverse_lookup_english(model, query, tokenizer, glove_vectors)
        print("Top predictions:")
        for word, score in results:
            print(f"  {word:20s} (similarity: {score:.4f})")
    
    if W is not None and fasttext_vectors is not None:
        print("\n" + "-"*70)
        print("--- German Reverse Dictionary (Cross-Lingual) ---")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = reverse_lookup_german(
                model, query, tokenizer, W, fasttext_vectors
            )
            print("Top predictions (German):")
            for word, score in results:
                print(f"  {word:20s} (similarity: {score:.4f})")


def main():
    """Main execution pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Train and run the Cross-Lingual Reverse Dictionary"
    )
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Train the model (requires OPTED-Dictionary.csv)"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference only (requires pretrained model)"
    )
    parser.add_argument(
        "--model",
        default="best_reverse_dict_model.pth",
        help="Path to model checkpoint (for inference)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't plot training history"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  Cross-Lingual Reverse Dictionary (English-German)")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Stage 1: Model training
    if args.train:
        model, history, tokenizer, glove_vectors = train_model(plot=not args.no_plot)
        if model is None:
            return
    else:
        # Load pretrained model for inference
        if not os.path.exists(args.model):
            print(f"Error: Model checkpoint '{args.model}' not found.")
            print("Use --train flag to train a new model.")
            return
        
        print(f"Loading model from {args.model}...")
        model, tokenizer = load_for_inference(args.model, BERT_MODEL, DEVICE)
        glove_vectors = api.load(GLOVE_MODEL)
    
    # Stage 2: Word alignment (if enabled)
    collapsed_dict = None
    W = None
    fasttext_vectors = None
    
    if os.path.exists(ALIGNMENT_FILE):
        collapsed_dict = setup_word_alignment()
    else:
        print(f"\nNote: Alignment file '{ALIGNMENT_FILE}' not found.")
        print("Skipping cross-lingual mapping. English mode only.")
    
    # Stage 3: Cross-lingual mapping (if alignment available)
    if collapsed_dict:
        try:
            W, glove_vectors, fasttext_vectors = setup_cross_lingual_mapping(collapsed_dict)
        except Exception as e:
            print(f"Warning: Could not setup cross-lingual mapping: {e}")
    
    # Stage 4: Inference
    print("\n" + "="*70)
    print("  Running Inference")
    print("="*70)
    
    try:
        demo_inference(W, glove_vectors, fasttext_vectors, model, tokenizer)
    except Exception as e:
        print(f"Inference error: {e}")
    
    print("\n" + "="*70)
    print("  Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
