#!/usr/bin/env python3
"""
Test script for FinBERT model functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

def test_finbert_model():
    """Test FinBERT model with sample financial text"""
    
    print("🔍 Testing FinBERT model...")
    
    try:
        # Load the FinBERT model from Hugging Face
        model_name = "ProsusAI/finbert"
        print(f"📥 Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print("✅ Model loaded successfully!")
        
        # Test sentences
        test_sentences = [
            "The company's revenue increased by 15% this quarter, showing strong growth.",
            "The stock price dropped significantly after the disappointing earnings report.",
            "The market remained stable with no significant changes in trading volume.",
            "Apple's new product launch exceeded expectations and boosted investor confidence.",
            "The economic uncertainty has led to increased market volatility."
        ]
        
        print("\n📊 Testing sentiment analysis on sample sentences:")
        print("=" * 60)
        
        results = []
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n{i}. {sentence}")
            
            # Tokenize and predict
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get probabilities
            probs = predictions[0].numpy()
            labels = ['negative', 'neutral', 'positive']
            
            # Find the predicted label
            predicted_label = labels[np.argmax(probs)]
            confidence = np.max(probs)
            
            # Calculate sentiment score (positive - negative)
            sentiment_score = probs[2] - probs[0]  # positive - negative
            
            print(f"   Predicted: {predicted_label} (confidence: {confidence:.3f})")
            print(f"   Sentiment Score: {sentiment_score:.3f}")
            print(f"   Probabilities: Negative={probs[0]:.3f}, Neutral={probs[1]:.3f}, Positive={probs[2]:.3f}")
            
            results.append({
                'sentence': sentence,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'sentiment_score': sentiment_score,
                'negative_prob': probs[0],
                'neutral_prob': probs[1],
                'positive_prob': probs[2]
            })
        
        # Save results to CSV
        df = pd.DataFrame(results)
        output_file = "finbert_test_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n✅ FinBERT model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing FinBERT model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_finbert_model()
    if success:
        print("\n🎉 FinBERT is ready to use!")
    else:
        print("\n💥 FinBERT test failed!")
        sys.exit(1)
