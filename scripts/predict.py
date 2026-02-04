"""
Make Predictions - Predict match outcomes
"""

import pandas as pd
import yaml
import joblib
import json
from pathlib import Path
import argparse


def load_model(config):
    """Load trained model and metadata"""
    models_dir = Path(config['data']['models'])
    
    # Load model
    model_path = models_dir / 'best_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train model first!")
    
    model = joblib.load(model_path)
    
    # Load feature columns
    features_path = models_dir / 'feature_columns.json'
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)
    
    return model, feature_cols


def predict_match(home_team: str, away_team: str, model, feature_cols, config):
    """
    Predict outcome for a match
    
    Note: This is a simplified version. In reality, you'd need to:
    1. Load historical data
    2. Calculate features for the teams
    3. Create feature vector
    4. Make prediction
    
    For now, this shows the structure.
    """
    print(f"\nPredicting: {home_team} vs {away_team}")
    print("-" * 40)
    
    # TODO: Extract features from historical data
    # For demonstration, we'll create dummy features
    # In practice, you'd call feature engineering functions here
    
    # Create dummy feature vector (zeros for now)
    features = pd.DataFrame([[0] * len(feature_cols)], columns=feature_cols)
    
    # Make prediction
    probabilities = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]
    
    # Get class labels (H, D, A)
    classes = model.classes_
    
    # Create probability dictionary
    prob_dict = dict(zip(classes, probabilities))
    
    # Convert probabilities to odds (odds = 1 / probability)
    odds_dict = {k: 1/v if v > 0 else 999 for k, v in prob_dict.items()}
    
    # Display results
    print(f"\nPredicted outcome: {prediction}")
    print(f"\nProbabilities:")
    print(f"  Home Win (H): {prob_dict.get('H', 0):.1%}  →  Odds: {odds_dict.get('H', 0):.2f}")
    print(f"  Draw (D):     {prob_dict.get('D', 0):.1%}  →  Odds: {odds_dict.get('D', 0):.2f}")
    print(f"  Away Win (A): {prob_dict.get('A', 0):.1%}  →  Odds: {odds_dict.get('A', 0):.2f}")
    
    return {
        'prediction': prediction,
        'probabilities': prob_dict,
        'odds': odds_dict
    }


def main():
    """Main prediction script"""
    parser = argparse.ArgumentParser(description='Predict EPL match outcome')
    parser.add_argument('--home', type=str, required=True, help='Home team name')
    parser.add_argument('--away', type=str, required=True, help='Away team name')
    args = parser.parse_args()
    
    print("="*60)
    print("EPL BETTING APP - MATCH PREDICTION")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\nLoading model...")
    model, feature_cols = load_model(config)
    print(f"✓ Model loaded ({len(feature_cols)} features)")
    
    # Make prediction
    result = predict_match(args.home, args.away, model, feature_cols, config)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
