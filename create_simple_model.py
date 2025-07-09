#!/usr/bin/env python3
"""
Create a simple model that matches the API schema for demonstration purposes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_compatible_features(df):
    """Create features that match the API schema from available dataset columns"""
    
    # Map dataset columns to API expected features
    feature_mapping = {
        'avg_tx_amount': df['tx_amount'],  # Use current transaction as proxy for average
        'device_change_freq': 0.05,  # Default constant (could be computed if we had user history)
        'tx_hour': df['tx_hour'],
        'location_change_freq': df['distance_from_usual_location'] / 1000.0,  # Normalize to 0-1 range
        'is_new_device': df['is_new_device'],
        'transaction_count_24h': df['transaction_count_24h'],
        'time_since_last_tx': 2.5,  # Default constant (could be computed if we had timestamps)
        'tx_amount_to_balance_ratio': df['tx_amount_to_balance_ratio'],
        'ip_address_reputation': df['ip_address_reputation'],
        'is_weekend': df['is_weekend'],
        'transaction_velocity_10min': df['transaction_velocity_10min'],
        'country_change_flag': df['country_change_flag']
    }
    
    # Create DataFrame with mapped features
    features_df = pd.DataFrame()
    for api_feature, data_source in feature_mapping.items():
        if isinstance(data_source, (int, float)):
            # Constant value
            features_df[api_feature] = [data_source] * len(df)
        else:
            # Column from dataset
            features_df[api_feature] = data_source
    
    return features_df

def main():
    logger.info("Loading synthetic behavioral dataset...")
    df = pd.read_csv('files/synthetic_behavioral_dataset.csv')
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Create features matching API schema
    X = create_compatible_features(df)
    y = df['risk_label']  # Use categorical labels
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target distribution: {y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit a scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a simple Random Forest model
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Test the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Save the model in the expected format
    model_package = {
        'model': model,
        'model_type': 'RandomForestClassifier',
        'scaler': scaler,
        'features': list(X.columns),
        'feature_count': len(X.columns)
    }
    
    joblib.dump(model_package, 'model.pkl')
    logger.info("Model saved as model.pkl")
    
    # Test prediction with sample data
    logger.info("Testing prediction...")
    sample_features = X.iloc[0:1]
    sample_scaled = scaler.transform(sample_features)
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    
    logger.info(f"Sample prediction: {prediction}")
    logger.info(f"Sample probabilities: {probabilities}")
    
    return model_package

if __name__ == "__main__":
    main()