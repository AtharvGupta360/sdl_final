"""
CodeBERT Inference Script for Bug Detection

Load the trained model and perform inference on new code samples.

Usage:
    # Single code sample
    python inference.py --model_path ./trained_model/final_model --code "int divide(int a, int b) { return a / b; }"
    
    # From file
    python inference.py --model_path ./trained_model/final_model --file path/to/code.c
    
    # Interactive mode
    python inference.py --model_path ./trained_model/final_model --interactive
"""

import argparse
import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os


class BugDetector:
    """
    Bug detection inference class using trained CodeBERT model.
    """
    def __init__(self, model_path):
        """
        Initialize the bug detector with a trained model.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"📦 Loading model from {model_path}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load configuration
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.max_length = self.config.get('max_length', 512)
        else:
            self.max_length = 512
        
        print("✅ Model loaded successfully!")
        if os.path.exists(config_path):
            print(f"   - Test Accuracy: {self.config.get('test_accuracy', 'N/A'):.4f}")
            print(f"   - Test F1: {self.config.get('test_f1', 'N/A'):.4f}")
    
    def predict(self, code):
        """
        Predict whether code contains bugs.
        
        Args:
            code: Source code string
            
        Returns:
            dict with:
                - is_buggy: bool (True if buggy)
                - confidence: float (probability of being buggy, 0-1)
                - label: str ("buggy" or "clean")
        """
        # Tokenize input
        inputs = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        # Prepare result
        is_buggy = (prediction == 1)
        buggy_prob = probs[0][1].item()  # Probability of being buggy
        
        return {
            'is_buggy': is_buggy,
            'confidence': confidence,
            'buggy_probability': buggy_prob,
            'clean_probability': probs[0][0].item(),
            'label': 'buggy' if is_buggy else 'clean'
        }
    
    def predict_batch(self, code_samples):
        """
        Predict bugs for multiple code samples.
        
        Args:
            code_samples: List of source code strings
            
        Returns:
            List of prediction dicts
        """
        results = []
        for code in code_samples:
            result = self.predict(code)
            results.append(result)
        return results


def print_prediction(code, result):
    """Pretty print prediction results."""
    print("\n" + "=" * 70)
    print("📝 Code Sample:")
    print("-" * 70)
    print(code[:300] + ("..." if len(code) > 300 else ""))
    print("-" * 70)
    print("\n🔍 Prediction:")
    print(f"   Label: {result['label'].upper()}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Buggy Probability: {result['buggy_probability']:.2%}")
    print(f"   Clean Probability: {result['clean_probability']:.2%}")
    
    if result['is_buggy']:
        print("\n⚠️  WARNING: This code likely contains bugs!")
    else:
        print("\n✅ This code appears to be clean.")
    print("=" * 70)


def interactive_mode(detector):
    """Run interactive inference mode."""
    print("\n🎯 Interactive Bug Detection Mode")
    print("Enter code (type 'END' on a new line to finish, 'quit' to exit):\n")
    
    while True:
        lines = []
        print(">>> ", end="")
        
        while True:
            try:
                line = input()
                if line.strip().lower() == 'quit':
                    print("\n👋 Goodbye!")
                    return
                if line.strip() == 'END':
                    break
                lines.append(line)
            except EOFError:
                return
        
        if lines:
            code = '\n'.join(lines)
            result = detector.predict(code)
            print_prediction(code, result)
            print("\nEnter more code (type 'END' to finish, 'quit' to exit):\n")


def main():
    parser = argparse.ArgumentParser(description='CodeBERT Bug Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--code', type=str, help='Code string to analyze')
    parser.add_argument('--file', type=str, help='Path to code file to analyze')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BugDetector(args.model_path)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(detector)
        return
    
    # File mode
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ Error: File not found: {args.file}")
            return
        
        with open(args.file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        result = detector.predict(code)
        print_prediction(code, result)
        return
    
    # Direct code mode
    if args.code:
        result = detector.predict(args.code)
        print_prediction(args.code, result)
        return
    
    # No input provided
    print("❌ Error: Please provide --code, --file, or --interactive")
    print("Usage examples:")
    print('  python inference.py --model_path ./trained_model/final_model --code "int divide(int a, int b) { return a / b; }"')
    print('  python inference.py --model_path ./trained_model/final_model --file code.c')
    print('  python inference.py --model_path ./trained_model/final_model --interactive')


if __name__ == '__main__':
    main()
