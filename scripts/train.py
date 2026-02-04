"""
Train Model - Complete training pipeline
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_feature_data(config):
    """Load feature-engineered data"""
    features_path = f"{config['data']['features']}/model_ready.csv"
    
    if not Path(features_path).exists():
        print(f"ERROR: Features file not found at {features_path}")
        print("Please run feature engineering first!")
        return None
    
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} matches with features")
    return df


def prepare_training_data(df, target_col='FTR'):
    """
    Prepare features and target for training
    """
    # Select feature columns (exclude non-feature columns)
    exclude_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'FTHG', 'FTAG',
        'HTR', 'HTHG', 'HTAG', 'Referee'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove any columns with all NaN
    feature_cols = [col for col in feature_cols if not df[col].isna().all()]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle any remaining missing values
    X = X.fillna(0)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y, feature_cols


def train_model(X, y, config):
    """
    Train machine learning model
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Split data (temporal split - older data for training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state'],
        shuffle=False  # Keep temporal order
    )
    
    print(f"\nTraining set: {len(X_train)} matches")
    print(f"Test set: {len(X_test)} matches")
    
    # Choose algorithm
    algorithm = config['model']['algorithm']
    
    if algorithm == 'xgboost':
        print("\nTraining XGBoost Classifier...")
        model = XGBClassifier(**config['model']['params'], random_state=42)
    elif algorithm == 'random_forest':
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(**config['model']['params'], random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n✓ Training complete!")
    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    
    # Detailed classification report
    print("\n--- Test Set Performance ---")
    print(classification_report(y_test, test_pred))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 10 Important Features ---")
        print(feature_importance.head(10))
    
    return model, {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'classification_report': classification_report(y_test, test_pred, output_dict=True)
    }


def save_model(model, feature_cols, metrics, config):
    """
    Save trained model and metadata
    """
    models_dir = Path(config['data']['models'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / 'best_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save feature columns
    features_path = models_dir / 'feature_columns.json'
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    print(f"✓ Feature columns saved to {features_path}")
    
    # Save metrics
    metrics_path = models_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("EPL BETTING APP - MODEL TRAINING")
    print("="*60)
    
    # Load config
    config = load_config()
    print(f"\nUsing algorithm: {config['model']['algorithm']}")
    
    # Load data
    df = load_feature_data(config)
    if df is None:
        return
    
    # Prepare data
    X, y, feature_cols = prepare_training_data(df)
    
    # Train model
    model, metrics = train_model(X, y, config)
    
    # Save everything
    save_model(model, feature_cols, metrics, config)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTest Accuracy: {metrics['test_accuracy']:.1%}")
    print("\nYou can now use predict.py to make predictions!")


if __name__ == "__main__":
    main()
