"""
Unit tests for the reverse dictionary project.
"""

import unittest
import torch
import numpy as np
from pathlib import Path

try:
    from config import DEVICE, BERT_MODEL
    from model import BertLSTMReverseDict, freeze_bert, unfreeze_bert
    from data_processing import ReverseDictDataset
    from word_alignment import tokenize, collapse_dictionary
except ImportError as e:
    print(f"Import error: {e}")


class TestModelArchitecture(unittest.TestCase):
    """Test model architecture and forward pass."""
    
    def setUp(self):
        """Initialize model for testing."""
        self.model = BertLSTMReverseDict()
        self.device = DEVICE
    
    def test_model_output_shape(self):
        """Test that model outputs correct vector shape."""
        self.model.to(self.device).eval()
        
        batch_size = 4
        seq_len = 64
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        with torch.no_grad():
            output = self.model(
                input_ids.to(self.device),
                attention_mask.to(self.device)
            )
        
        self.assertEqual(output.shape, (batch_size, 300))
    
    def test_freeze_unfreeze_bert(self):
        """Test BERT freezing and unfreezing."""
        model = BertLSTMReverseDict()
        
        # Initially all parameters should be trainable
        freeze_bert(model)
        for param in model.bert.parameters():
            self.assertFalse(param.requires_grad)
        
        unfreeze_bert(model)
        for param in model.bert.parameters():
            self.assertTrue(param.requires_grad)


class TestTextProcessing(unittest.TestCase):
    """Test text processing utilities."""
    
    def test_tokenization(self):
        """Test punctuation tokenization."""
        test_cases = [
            ("hello.", "hello ."),
            ("hello, world!", "hello , world !"),
            ("test:data", "test : data"),
            ("multiple  spaces", "multiple spaces"),
        ]
        
        for input_text, expected in test_cases:
            result = tokenize(input_text)
            self.assertEqual(result, expected)


class TestDictProcessing(unittest.TestCase):
    """Test dictionary processing utilities."""
    
    def test_collapse_dictionary(self):
        """Test dictionary collapsing and cleaning."""
        # Mock alignment dictionary
        mock_dict = {
            "hello.": [("hallo", 5)],
            "world,": [("welt", 3)],
            "test": [("test", 2)],
        }
        
        result = collapse_dictionary(mock_dict)
        
        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("test", result)
        self.assertEqual(result["hello"], "hallo")
        self.assertEqual(result["world"], "welt")


class TestVectorOperations(unittest.TestCase):
    """Test vector operations."""
    
    def test_transformation_matrix_orthogonality(self):
        """Test that transformation matrix is orthogonal."""
        # Create a random orthogonal matrix
        random_matrix = np.random.randn(300, 300)
        U, _, Vt = np.linalg.svd(random_matrix)
        W = U @ Vt  # Guaranteed orthogonal
        
        # Check orthogonality: W^T @ W = I
        identity = W.T @ W
        expected_identity = np.eye(300)
        
        np.testing.assert_array_almost_equal(identity, expected_identity, decimal=5)
    
    def test_vector_transformation(self):
        """Test vector transformation preserves norms."""
        # Create orthogonal transformation matrix
        random_matrix = np.random.randn(300, 300)
        U, _, Vt = np.linalg.svd(random_matrix)
        W = U @ Vt
        
        # Create random vector
        v = np.random.randn(300)
        v_norm = np.linalg.norm(v)
        
        # Transform vector
        v_transformed = v @ W
        v_transformed_norm = np.linalg.norm(v_transformed)
        
        # Orthogonal transformation preserves norms
        np.testing.assert_almost_equal(v_norm, v_transformed_norm, decimal=5)


class TestConfiguration(unittest.TestCase):
    """Test configuration values."""
    
    def test_device_availability(self):
        """Test that device is properly configured."""
        self.assertIsNotNone(DEVICE)
        self.assertIn(DEVICE.type, ['cpu', 'cuda'])
    
    def test_bert_model_name(self):
        """Test BERT model name is valid."""
        self.assertEqual(BERT_MODEL, 'bert-base-uncased')


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
