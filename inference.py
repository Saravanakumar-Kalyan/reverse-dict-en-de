"""
Inference module for reverse dictionary lookups (English and German).
"""

import torch
import numpy as np
from transformers import BertTokenizer

from config import DEVICE, TOP_K_PREDICTIONS, BERT_MODEL
from model import load_model_for_inference


def reverse_lookup_english(model, definition, tokenizer, glove_vectors, top_k=TOP_K_PREDICTIONS):
    """
    English reverse dictionary lookup: definition -> English word.
    
    Takes an English definition and returns the most similar English words
    in the GloVe vocabulary.
    
    Args:
        model: Trained BertLSTMReverseDict model
        definition: English definition string
        tokenizer: BERT tokenizer
        glove_vectors: GloVe word vectors
        top_k: Number of results to return
        
    Returns:
        List of (word, similarity_score) tuples
    """
    model.eval()
    
    inputs = tokenizer(
        definition, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(DEVICE)
    
    with torch.no_grad():
        pred_vec = model(
            inputs['input_ids'],
            inputs['attention_mask']
        ).cpu().numpy()[0]
    
    # Find most similar words in GloVe vocabulary
    similar_words = glove_vectors.most_similar(positive=[pred_vec], topn=top_k)
    
    return similar_words


def reverse_lookup_german(model, definition, tokenizer, transformation_matrix, 
                         de_vectors, top_k=TOP_K_PREDICTIONS):
    """
    Cross-lingual reverse dictionary lookup: English definition -> German word.
    
    Process:
    1. Generate English concept vector from definition using model
    2. Transform English vector to German space using transformation matrix W
    3. Find most similar words in German FastText vocabulary
    
    Args:
        model: Trained BertLSTMReverseDict model
        definition: English definition string
        tokenizer: BERT tokenizer
        transformation_matrix: Orthogonal Procrustes transformation matrix (W)
        de_vectors: German FastText word vectors
        top_k: Number of results to return
        
    Returns:
        List of (word, similarity_score) tuples in German
    """
    model.eval()
    
    # Generate English vector from definition
    inputs = tokenizer(
        definition,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(DEVICE)
    
    with torch.no_grad():
        en_pred_vec = model(
            inputs['input_ids'],
            inputs['attention_mask']
        ).cpu().numpy()[0]
    
    # Transform to German space
    de_pred_vec = np.dot(en_pred_vec, transformation_matrix)
    
    # Find most similar words in German vocabulary
    similar_words = de_vectors.most_similar(positive=[de_pred_vec], topn=top_k)
    
    return similar_words


def load_for_inference(model_checkpoint, model_name=BERT_MODEL, device=DEVICE):
    """
    Load everything needed for inference.
    
    Args:
        model_checkpoint: Path to saved model weights
        model_name: Name of BERT model
        device: Device to use
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model = load_model_for_inference(model_checkpoint, device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


def interactive_reverse_lookup(model, tokenizer, glove_vectors, de_vectors=None, 
                              transformation_matrix=None, language='en'):
    """
    Interactive mode for reverse dictionary lookups.
    
    Args:
        model: Trained model
        tokenizer: BERT tokenizer
        glove_vectors: English vectors
        de_vectors: German vectors (optional for German lookups)
        transformation_matrix: Transformation matrix W (optional for German lookups)
        language: 'en' for English, 'de' for German (cross-lingual)
    """
    print(f"\n{'='*60}")
    print(f"  Reverse Dictionary Lookup ({language.upper()})")
    print(f"{'='*60}")
    print("Enter an English definition to find the corresponding word.")
    print("Type 'quit' to exit.\n")
    
    while True:
        definition = input("Definition: ").strip()
        
        if definition.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not definition:
            print("Please enter a non-empty definition.\n")
            continue
        
        try:
            if language == 'en':
                results = reverse_lookup_english(
                    model, definition, tokenizer, glove_vectors
                )
                print(f"\nTop predictions (English):")
            else:  # language == 'de'
                if de_vectors is None or transformation_matrix is None:
                    print("Error: German vectors and transformation matrix required.\n")
                    continue
                
                results = reverse_lookup_german(
                    model, definition, tokenizer, transformation_matrix,
                    de_vectors
                )
                print(f"\nTop predictions (German):")
            
            for i, (word, score) in enumerate(results, 1):
                print(f"  {i}. {word:20s} (similarity: {score:.4f})")
            print()
            
        except Exception as e:
            print(f"Error: {e}\n")
