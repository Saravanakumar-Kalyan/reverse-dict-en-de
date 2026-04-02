"""
Training logic for the BERT-LSTM Reverse Dictionary model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from transformers import BertTokenizer

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, FINE_TUNE_LR, EPOCHS, 
    BERT_FREEZE_UNTIL_EPOCH, SAVE_PATH, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    BERT_MODEL, GLOVE_MODEL
)
from model import BertLSTMReverseDict, freeze_bert, unfreeze_bert
from data_processing import ReverseDictDataset, load_tokenizer_and_vectors


def run_training(csv_path):
    """
    Complete training pipeline for reverse dictionary model.
    
    Training Strategy:
    1. Epochs 1-4: BERT frozen, only LSTM and linear layers train
    2. Epochs 5+: BERT unfrozen with lower learning rate for fine-tuning
    
    Args:
        csv_path: Path to dictionary CSV file
        
    Returns:
        Tuple of (trained_model, history_dict, tokenizer, glove_vectors)
    """
    
    # Load tokenizer and vectors
    tokenizer, glove_vectors = load_tokenizer_and_vectors(BERT_MODEL)
    
    # Create dataset and data loaders
    dataset = ReverseDictDataset(csv_path, tokenizer, glove_vectors)
    train_size = int(0.8 * len(dataset))
    val_ds, train_ds = random_split(
        dataset, 
        [len(dataset) - train_size, train_size]
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = BertLSTMReverseDict().to(DEVICE)
    freeze_bert(model)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE
    )
    criterion = nn.MSELoss()
    
    # Training loop
    best_v_loss = float('inf')
    history = {'train': [], 'val': []}
    
    print(f"\nStarting training on {DEVICE}...")
    print(f"Total epochs: {EPOCHS}, Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    for epoch in range(EPOCHS):
        # Unfreeze BERT at specified epoch
        if epoch == BERT_FREEZE_UNTIL_EPOCH:
            print(f"\n--- Unfreezing BERT at epoch {epoch} ---")
            unfreeze_bert(model)
            optimizer.param_groups[0]['lr'] = FINE_TUNE_LR
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train'].append(avg_train_loss)
        history['val'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if avg_val_loss < best_v_loss:
            best_v_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")
    
    print(f"\nTraining complete! Best model saved to {SAVE_PATH}")
    return model, history, tokenizer, glove_vectors


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss over epochs.
    
    Args:
        history: Dictionary with 'train' and 'val' loss lists
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Train Loss', marker='o')
    plt.plot(history['val'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
