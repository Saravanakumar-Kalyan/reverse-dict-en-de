"""
Model architecture for the Cross-Lingual Reverse Dictionary.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from config import BERT_MODEL, DEVICE


class BertLSTMReverseDict(nn.Module):
    """
    Hybrid BERT-LSTM model for reverse dictionary lookup.
    
    Architecture:
    - BERT: Feature extractor to understand contextual meaning of definitions
    - Bi-LSTM: Captures directional dependencies in BERT outputs
    - Linear Layer: Projects 512-dim hidden state to 300-dim GloVe embedding
    """
    
    def __init__(self, dropout_rate=0.4):
        super(BertLSTMReverseDict, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, 300)  # 256*2 bidirectional = 512 input

    def forward(self, ids, mask):
        """
        Forward pass through the model.
        
        Args:
            ids: Input token IDs from BERT tokenizer
            mask: Attention mask for padding tokens
            
        Returns:
            Predicted 300-dimensional word vector
        """
        bert_out = self.bert(input_ids=ids, attention_mask=mask)
        _, (h_n, _) = self.lstm(bert_out.last_hidden_state)
        
        # Concatenate final forward and backward hidden states from both layers
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        return self.fc(self.dropout(hidden))


def load_model_for_inference(checkpoint_path, device=DEVICE):
    """
    Load a pre-trained model checkpoint for inference.
    
    Args:
        checkpoint_path: Path to saved model weights
        device: Device to load model on (cpu/cuda)
        
    Returns:
        Model in evaluation mode
    """
    model = BertLSTMReverseDict().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def freeze_bert(model):
    """Freeze all BERT parameters to prevent weight destruction."""
    for param in model.bert.parameters():
        param.requires_grad = False


def unfreeze_bert(model):
    """Unfreeze all BERT parameters for fine-tuning."""
    for param in model.bert.parameters():
        param.requires_grad = True
