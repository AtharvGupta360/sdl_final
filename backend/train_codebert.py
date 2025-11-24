"""
CodeBERT Fine-tuning for Bug Detection / Fault Localization

This script fine-tunes microsoft/codebert-base on the CodeXGLUE Defect Detection dataset
to predict whether code contains bugs.

Dataset: CodeXGLUE Defect Detection (devign)
- Binary classification: buggy (1) vs clean (0)
- ~27k C code samples with labels

Usage:
    python train_codebert.py --epochs 3 --batch_size 16 --learning_rate 2e-5
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from tqdm import tqdm


class CodeDefectDataset(Dataset):
    """
    PyTorch Dataset for code defect detection.
    Handles tokenization and label preparation.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get code and label
        code = item['func']  # Function code
        label = item['target']  # 0 = clean, 1 = buggy
        
        # Tokenize code
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data(tokenizer, max_length=512):
    """
    Load CodeXGLUE Defect Detection dataset and prepare train/val/test splits.
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print("📦 Loading CodeXGLUE Defect Detection dataset...")
    
    # Load dataset from HuggingFace
    # Dataset: code_x_glue_cc_defect_detection
    dataset = load_dataset("code_x_glue_cc_defect_detection", "defect_detection")
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Train samples: {len(dataset['train'])}")
    print(f"   - Validation samples: {len(dataset['validation'])}")
    print(f"   - Test samples: {len(dataset['test'])}")
    
    # Create PyTorch datasets
    train_dataset = CodeDefectDataset(dataset['train'], tokenizer, max_length)
    val_dataset = CodeDefectDataset(dataset['validation'], tokenizer, max_length)
    test_dataset = CodeDefectDataset(dataset['test'], tokenizer, max_length)
    
    return train_dataset, val_dataset, test_dataset


def compute_metrics(pred):
    """
    Compute evaluation metrics for binary classification.
    
    Args:
        pred: Predictions from Trainer
        
    Returns:
        dict with accuracy, precision, recall, f1
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def train_model(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("\n📝 Loading CodeBERT tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Load and prepare datasets
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        tokenizer, 
        max_length=args.max_length
    )
    
    # Load pre-trained CodeBERT model
    print("\n🤖 Loading CodeBERT model...")
    model = RobertaForSequenceClassification.from_pretrained(
        'microsoft/codebert-base',
        num_labels=2  # Binary classification: clean (0) vs buggy (1)
    )
    model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
        eval_strategy='epoch',  # Evaluate every epoch
        save_strategy='epoch',  # Save checkpoint every epoch
        save_total_limit=2,  # Keep only 2 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        report_to='none',  # Disable wandb/tensorboard
        seed=42
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("\n🔥 Starting training...")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Max sequence length: {args.max_length}")
    print()
    
    train_result = trainer.train()
    
    # Save training metrics
    print("\n💾 Saving training metrics...")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    print("\n📊 Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    print(f"   - Validation Accuracy: {val_metrics['eval_accuracy']:.4f}")
    print(f"   - Validation F1: {val_metrics['eval_f1']:.4f}")
    print(f"   - Validation Precision: {val_metrics['eval_precision']:.4f}")
    print(f"   - Validation Recall: {val_metrics['eval_recall']:.4f}")
    trainer.save_metrics("eval", val_metrics)
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset)
    print(f"   - Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"   - Test F1: {test_metrics['eval_f1']:.4f}")
    print(f"   - Test Precision: {test_metrics['eval_precision']:.4f}")
    print(f"   - Test Recall: {test_metrics['eval_recall']:.4f}")
    trainer.save_metrics("test", test_metrics)
    
    # Save the final model
    print(f"\n💾 Saving final model to {args.output_dir}/final_model...")
    final_model_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save training configuration
    config = {
        'model_name': 'microsoft/codebert-base',
        'task': 'binary_classification',
        'num_labels': 2,
        'max_length': args.max_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'test_accuracy': float(test_metrics['eval_accuracy']),
        'test_f1': float(test_metrics['eval_f1']),
        'test_precision': float(test_metrics['eval_precision']),
        'test_recall': float(test_metrics['eval_recall'])
    }
    
    with open(f"{final_model_path}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {final_model_path}")
    print(f"   Ready to use in your backend!")
    
    return trainer, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CodeBERT for bug detection')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--output_dir', type=str, default='./trained_model', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🐛 CodeBERT Fine-tuning for Bug Detection")
    print("=" * 70)
    
    train_model(args)


if __name__ == '__main__':
    main()
