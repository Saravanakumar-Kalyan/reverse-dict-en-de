"""
Example script demonstrating how to use the reverse dictionary.
"""

import os
import torch
import numpy as np
import gensim.downloader as api

from config import DEVICE, BERT_MODEL, GLOVE_MODEL, FASTTEXT_MODEL
from model import load_model_for_inference
from inference import reverse_lookup_english, reverse_lookup_german, load_for_inference
from vector_mapping import load_transformation_matrix


def example_english_only():
    """Example: English reverse dictionary lookup only."""
    
    print("\n" + "="*70)
    print("  Example 1: English-Only Reverse Dictionary")
    print("="*70)
    
    # Load model and vectors
    model, tokenizer = load_for_inference("best_reverse_dict_model.pth", BERT_MODEL, DEVICE)
    glove_vectors = api.load(GLOVE_MODEL)
    
    # Test definitions
    definitions = [
        "A large fruit with a hard green skin and red juicy flesh",
        "A feeling of great pleasure or satisfaction",
        "A person who studies science",
        "A small furry animal that barks",
    ]
    
    print("\nEnglish Reverse Dictionary Lookups:")
    print("-" * 70)
    
    for definition in definitions:
        print(f"\nQuery: '{definition}'")
        print("Top predictions:")
        
        results = reverse_lookup_english(
            model, definition, tokenizer, glove_vectors, top_k=5
        )
        
        for i, (word, score) in enumerate(results, 1):
            print(f"  {i}. {word:20s} (similarity: {score:.4f})")


def example_cross_lingual():
    """Example: Cross-lingual English-to-German reverse dictionary."""
    
    print("\n" + "="*70)
    print("  Example 2: Cross-Lingual Reverse Dictionary")
    print("="*70)
    
    # Check if required files exist
    if not os.path.exists("transformation_matrix.npy"):
        print("\nError: Transformation matrix 'transformation_matrix.npy' not found.")
        print("Please generate it by running the full training pipeline first.")
        return
    
    # Load model, tokenizer, and vectors
    model, tokenizer = load_for_inference("best_reverse_dict_model.pth", BERT_MODEL, DEVICE)
    glove_vectors = api.load(GLOVE_MODEL)
    de_vectors = api.load(FASTTEXT_MODEL)
    
    # Load transformation matrix
    W = load_transformation_matrix("transformation_matrix.npy")
    
    # Test definitions
    definitions = [
        "A large fruit with a hard green skin and red juicy flesh",
        "A feeling of great pleasure or satisfaction",
    ]
    
    print("\nCross-Lingual (English -> German) Reverse Dictionary Lookups:")
    print("-" * 70)
    
    for definition in definitions:
        print(f"\nEnglish Query: '{definition}'")
        
        # English results
        en_results = reverse_lookup_english(
            model, definition, tokenizer, glove_vectors, top_k=3
        )
        print("English predictions:")
        for i, (word, score) in enumerate(en_results, 1):
            print(f"  {i}. {word:20s} (similarity: {score:.4f})")
        
        # German results
        de_results = reverse_lookup_german(
            model, definition, tokenizer, W, de_vectors, top_k=3
        )
        print("German predictions:")
        for i, (word, score) in enumerate(de_results, 1):
            print(f"  {i}. {word:20s} (similarity: {score:.4f})")


def example_custom_inference():
    """Example: Custom inference with your own definitions."""
    
    print("\n" + "="*70)
    print("  Example 3: Interactive Mode")
    print("="*70)
    
    # Load components
    model, tokenizer = load_for_inference("best_reverse_dict_model.pth", BERT_MODEL, DEVICE)
    glove_vectors = api.load(GLOVE_MODEL)
    
    print("\nEnter definitions to find corresponding words.")
    print("Type 'quit' to exit.\n")
    
    while True:
        definition = input("Enter definition (or 'quit'): ").strip()
        
        if definition.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not definition:
            print("Please enter a non-empty definition.\n")
            continue
        
        try:
            results = reverse_lookup_english(
                model, definition, tokenizer, glove_vectors, top_k=5
            )
            
            print("Top predictions:")
            for i, (word, score) in enumerate(results, 1):
                print(f"  {i}. {word:20s} (similarity: {score:.4f})")
            print()
            
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Run examples."""
    
    print("\n" + "="*70)
    print("  Cross-Lingual Reverse Dictionary - Usage Examples")
    print("="*70)
    print(f"Device: {DEVICE}\n")
    
    # Check if model exists
    if not os.path.exists("best_reverse_dict_model.pth"):
        print("Error: Pre-trained model 'best_reverse_dict_model.pth' not found.")
        print("Please train the model first using: python main.py --train")
        return
    
    # Run examples
    try:
        example_english_only()
        print("\n")
        
        # Try cross-lingual example (may fail if transformation matrix not available)
        try:
            example_cross_lingual()
            print("\n")
        except FileNotFoundError:
            print("\nNote: Skipping cross-lingual example (require transformation matrix)")
        
        # Interactive example
        example_custom_inference()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
