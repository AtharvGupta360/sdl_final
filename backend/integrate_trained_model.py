"""
Integration Module: Use Trained CodeBERT in Backend

This module replaces the heuristic scoring in ai_model.py with the actual trained model.

Usage:
    1. Train the model: python train_codebert.py --epochs 3 --batch_size 16
    2. Update TRAINED_MODEL_PATH below to point to your trained model
    3. Replace the AIModel class in ai_model.py with this implementation
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sqlalchemy.orm import Session
import hashlib


# ============================================================================
# CONFIGURATION: Update this path after training
# ============================================================================
TRAINED_MODEL_PATH = "./trained_model/final_model"


class AIModel:
    """
    AI Model using trained CodeBERT for fault localization.
    
    This class uses the fine-tuned CodeBERT model to predict bug probabilities
    for each line of code in a repository.
    """
    
    def __init__(self, db: Session):
        """Initialize the trained CodeBERT model."""
        self.db = db
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Loading trained CodeBERT model from {TRAINED_MODEL_PATH}...")
        
        try:
            # Load tokenizer and model
            self.tokenizer = RobertaTokenizer.from_pretrained(TRAINED_MODEL_PATH)
            self.model = RobertaForSequenceClassification.from_pretrained(TRAINED_MODEL_PATH)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Trained model loaded successfully!")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            print(f"⚠️  Falling back to heuristic scoring...")
            self.model = None
    
    def predict(self, candidates: List[Dict]) -> List[Dict]:
        """
        Predict bug probabilities for candidate lines.
        
        Args:
            candidates: List of dicts with 'code', 'line_number', 'file_path', 'features'
            
        Returns:
            List of ranked candidates with probabilities
        """
        if not candidates:
            return []
        
        # Use trained model if available, otherwise fall back to heuristics
        if self.model is not None:
            scores = self._trained_model_predict(candidates)
        else:
            scores = self._heuristic_score(candidates)
        
        # Add probabilities to candidates
        for i, candidate in enumerate(candidates):
            candidate['probability'] = float(scores[i])
        
        # Sort by probability (highest first)
        ranked = sorted(candidates, key=lambda x: x['probability'], reverse=True)
        
        return ranked
    
    def _trained_model_predict(self, candidates: List[Dict]) -> np.ndarray:
        """
        Use the trained CodeBERT model to predict bug probabilities.
        
        Args:
            candidates: List of candidate dicts with 'code' field
            
        Returns:
            numpy array of probabilities (0-1)
        """
        scores = []
        
        # Process in batches for efficiency
        batch_size = 32
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_codes = [c.get('code', '') for c in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_codes,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get probability of class 1 (buggy)
                buggy_probs = probs[:, 1].cpu().numpy()
                scores.extend(buggy_probs)
        
        # Adjust scores based on additional features (stack trace, complexity, etc.)
        adjusted_scores = self._adjust_scores_with_features(scores, candidates)
        
        return np.array(adjusted_scores)
    
    def _adjust_scores_with_features(self, base_scores: List[float], candidates: List[Dict]) -> List[float]:
        """
        Adjust model predictions using additional features.
        
        The trained model gives base probability, but we can boost it using:
        - Stack trace information (most important)
        - Code complexity
        - Recent changes
        
        Args:
            base_scores: Base probabilities from the model
            candidates: Candidate dicts with features
            
        Returns:
            Adjusted probabilities
        """
        adjusted = []
        
        for score, candidate in zip(base_scores, candidates):
            features = candidate.get('features', {})
            
            # Start with model prediction
            final_score = score
            
            # Stack trace lines are HIGHLY suspicious (big boost)
            if features.get('distance_from_error', 1.0) < 0.1:
                final_score = min(final_score + 0.30, 0.98)  # Big boost for stack trace
            
            # High complexity boost
            complexity = features.get('cyclomatic_complexity', 0.0)
            if complexity > 5:
                final_score = min(final_score + 0.10, 0.98)
            
            # Recent changes boost
            churn = features.get('code_churn', 0.0)
            if churn > 0.5:
                final_score = min(final_score + 0.08, 0.98)
            
            adjusted.append(final_score)
        
        return adjusted
    
    def _heuristic_score(self, candidates: List[Dict]) -> np.ndarray:
        """
        Fallback heuristic scoring if trained model not available.
        (Keep the existing heuristic implementation as backup)
        """
        scores = []
        
        for candidate in candidates:
            features = candidate.get('features', {})
            code = candidate.get('code', '').lower()
            
            # Base score - much lower
            score = 0.15
            
            # Stack trace lines are HIGHLY suspicious (massive boost)
            if features.get('distance_from_error', 1.0) < 0.1:
                score += 0.75
            
            # High complexity is suspicious
            complexity_boost = features.get('cyclomatic_complexity', 0.0) * 0.15
            score += complexity_boost
            
            # Recent changes are suspicious
            churn_boost = features.get('code_churn', 0.0) * 0.1
            score += churn_boost
            
            # High-risk bug patterns
            high_risk_patterns = [
                ('==', 0.12), ('!=', 0.12), ('- ', 0.10), ('+ ', 0.08),
                ('null', 0.10), ('none', 0.10), ('undefined', 0.10),
            ]
            
            # Medium-risk patterns
            medium_risk_patterns = [
                ('if', 0.04), ('return', 0.04), ('while', 0.03), ('for', 0.03),
            ]
            
            for pattern, boost in high_risk_patterns:
                if pattern in code:
                    score += boost
            
            if score < 0.4:
                for pattern, boost in medium_risk_patterns:
                    if pattern in code:
                        score += boost
            
            scores.append(min(score, 0.95))
        
        return np.array(scores)


# ============================================================================
# INSTRUCTIONS FOR INTEGRATION
# ============================================================================
"""
📋 How to integrate the trained model into your backend:

1. TRAIN THE MODEL:
   cd C:\Users\gupta\OneDrive\Desktop\sdl_final\backend
   python train_codebert.py --epochs 3 --batch_size 16 --output_dir ./trained_model

2. VERIFY MODEL FILES:
   Check that these files exist:
   - ./trained_model/final_model/pytorch_model.bin
   - ./trained_model/final_model/config.json
   - ./trained_model/final_model/tokenizer_config.json

3. UPDATE ai_model.py:
   Replace the AIModel class in ai_model.py with the class above.
   
   OR use this code snippet in ai_model.py at the top:
   
   ```python
   from integrate_trained_model import AIModel
   ```
   
   And remove/comment out the existing AIModel class.

4. RESTART THE BACKEND:
   python main.py

5. TEST:
   The frontend will now use the trained model for predictions!
   You should see much more accurate bug detection.

6. MONITORING:
   Watch the console when the backend starts:
   - "✅ Trained model loaded successfully!" = Using trained model
   - "⚠️  Falling back to heuristic scoring..." = Using heuristics (model not found)

NOTE: Training takes ~30-60 minutes on GPU, 2-4 hours on CPU for 3 epochs.
"""
