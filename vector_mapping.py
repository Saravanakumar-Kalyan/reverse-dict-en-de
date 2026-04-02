"""
Cross-lingual vector space mapping using Orthogonal Procrustes.
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from typing import Dict, Tuple


def learn_vector_mapping(clean_dict: Dict[str, str], 
                        en_vectors, de_vectors) -> np.ndarray:
    """
    Learn transformation matrix to map English vectors to German vector space.
    
    Uses Orthogonal Procrustes to find the optimal rotation matrix W
    that aligns the English and German word embeddings.
    
    Mathematical approach:
    - Collect aligned word pairs from both vector spaces
    - Compute W = argmin ||XW - Y||_F (Frobenius norm)
    - W is an orthogonal matrix (preserves distances and angles)
    
    Args:
        clean_dict: Mapping of English words to German words
        en_vectors: English word embeddings (GloVe)
        de_vectors: German word embeddings (FastText)
        
    Returns:
        Transformation matrix W (300x300 for word vectors)
    """
    X, Y = [], []
    matched_pairs = 0
    
    print("Matching word pairs across vector spaces...")
    
    for en_word, de_word in clean_dict.items():
        if en_word in en_vectors and de_word in de_vectors:
            X.append(en_vectors[en_word])
            Y.append(de_vectors[de_word])
            matched_pairs += 1
    
    print(f"Successfully matched {matched_pairs} words across both models.")
    
    if matched_pairs < 50:
        raise ValueError(
            f"Not enough matched pairs ({matched_pairs}). "
            "Check if vectors are properly loaded."
        )
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Compute orthogonal Procrustes solution
    W, residual = orthogonal_procrustes(X, Y)
    
    print(f"Transformation matrix learned with residual: {residual:.6f}")
    print(f"Matrix shape: {W.shape} (square orthogonal matrix)")
    
    return W


def save_transformation_matrix(W: np.ndarray, filepath: str):
    """
    Save transformation matrix to file.
    
    Args:
        W: Transformation matrix
        filepath: Path to save the matrix
    """
    np.save(filepath, W)
    print(f"Transformation matrix saved to {filepath}")


def load_transformation_matrix(filepath: str) -> np.ndarray:
    """
    Load transformation matrix from file.
    
    Args:
        filepath: Path to saved matrix file
        
    Returns:
        Loaded transformation matrix
    """
    W = np.load(filepath)
    print(f"Transformation matrix loaded from {filepath}")
    return W


def transform_vector(vector: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Transform a single vector from English to German space.
    
    Args:
        vector: English word vector (300-dim)
        W: Transformation matrix
        
    Returns:
        Transformed vector in German space (300-dim)
    """
    return np.dot(vector, W)


def transform_vectors(vectors: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Transform multiple vectors from English to German space.
    
    Args:
        vectors: Array of English word vectors (N × 300)
        W: Transformation matrix (300 × 300)
        
    Returns:
        Array of transformed vectors in German space (N × 300)
    """
    return np.dot(vectors, W)


def verify_transformation_quality(en_vectors, de_vectors, W: np.ndarray,
                                 sample_words: list = None) -> float:
    """
    Verify quality of transformation by computing similarity preservation.
    
    Checks if the Euclidean distances between word pairs are preserved
    after transformation.
    
    Args:
        en_vectors: English word vectors
        de_vectors: German word vectors
        W: Transformation matrix
        sample_words: List of English words to test (uses first 100 if None)
        
    Returns:
        Average correlation of distances before and after transformation
    """
    if sample_words is None:
        sample_words = list(en_vectors.index_to_key[:100])
    
    distances_before = []
    distances_after = []
    
    for i in range(min(len(sample_words) - 1, 50)):
        word1 = sample_words[i]
        word2 = sample_words[i + 1]
        
        if word1 in en_vectors and word2 in en_vectors:
            # Original distance in English space
            en_vec1 = en_vectors[word1]
            en_vec2 = en_vectors[word2]
            dist_en = np.linalg.norm(en_vec1 - en_vec2)
            distances_before.append(dist_en)
            
            # Distance after transformation
            de_vec1 = transform_vector(en_vec1, W)
            de_vec2 = transform_vector(en_vec2, W)
            dist_de = np.linalg.norm(de_vec1 - de_vec2)
            distances_after.append(dist_de)
    
    # Compute correlation
    if len(distances_before) > 0:
        correlation = np.corrcoef(distances_before, distances_after)[0, 1]
        print(f"Distance preservation correlation: {correlation:.4f}")
        return correlation
    
    return 0.0
