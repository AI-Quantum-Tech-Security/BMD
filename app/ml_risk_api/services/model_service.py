"""
Model service for loading and serving the XGBoost risk assessment model
"""

import joblib
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling ML model operations"""

    def __init__(self, model_path: str = "files/enhanced_model.pkl"):
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = None
        self.feature_metadata = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Load the trained XGBoost model and preprocessing components"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading model from: {self.model_path}")
            self.model_package = joblib.load(self.model_path)

            # Extract components
            self.model = self.model_package['model']
            self.scaler = self.model_package.get('scaler')
            self.label_encoder = self.model_package['label_encoder']
            self.features = self.model_package['features']
            self.feature_metadata = self.model_package.get('feature_metadata', {})

            logger.info(f"Model loaded successfully:")
            logger.info(f"  Type: {self.model_package.get('model_type', 'Unknown')}")
            logger.info(f"  Features: {len(self.features)}")
            logger.info(f"  Classes: {list(self.model.classes_)}")

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.is_loaded and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        training_metrics = self.model_package.get('training_metrics', {})

        return {
            "model_type": self.model_package.get('model_type', 'XGBoost'),
            "model_version": "1.0",
            "training_date": self.model_package.get('training_date', datetime.now().isoformat()),
            "feature_count": len(self.features),
            "classes": list(self.model.classes_),
            "training_metrics": training_metrics,
            "features": self.features
        }

    def _preprocess_features(self, input_features: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input features to match training format

        Args:
            input_features: Raw input features from API request

        Returns:
            Preprocessed feature DataFrame ready for prediction
        """
        try:
            # Create DataFrame from input
            df = pd.DataFrame([input_features])

            # Feature engineering (recreate training transformations)
            df = self._apply_feature_engineering(df)

            # Handle categorical features (one-hot encoding)
            df = self._handle_categorical_features(df)

            # Ensure all required features are present
            df = self._align_features(df)

            # Apply scaling if scaler is available
            if self.scaler is not None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])

            return df

        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}")
            raise ValueError(f"Feature preprocessing error: {str(e)}")

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations matching training pipeline"""

        # Transaction amount interactions
        if 'tx_amount' in df.columns and 'avg_tx_amount' in df.columns:
            df['tx_amount_ratio'] = df['tx_amount'] / (df['avg_tx_amount'] + 1e-8)
            df['tx_amount_deviation'] = abs(df['tx_amount'] - df['avg_tx_amount'])

        # Behavioral interactions
        if 'session_duration' in df.columns and 'transaction_count_24h' in df.columns:
            df['session_tx_intensity'] = df['transaction_count_24h'] / (df['session_duration'] + 1)

        # Risk combinations
        if 'device_change_freq' in df.columns and 'location_change_freq' in df.columns:
            df['behavior_volatility'] = df['device_change_freq'] + df['location_change_freq']
            df['mobility_risk'] = df['device_change_freq'] * df['location_change_freq']

        # Network risk
        if 'ip_address_reputation' in df.columns and 'vpn_proxy_flag' in df.columns:
            df['network_risk_combined'] = df['ip_address_reputation'] * (1 + df['vpn_proxy_flag'])

        # Behavioral consistency
        if 'typing_pattern_similarity' in df.columns and 'mouse_movement_similarity' in df.columns:
            df['behavioral_consistency'] = (
                                                   df['typing_pattern_similarity'] + df['mouse_movement_similarity']
                                           ) / 2

        return df

    def _handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical features with one-hot encoding"""

        categorical_features = ['merchant_id', 'device_type', 'location']

        for col in categorical_features:
            if col in df.columns:
                # Simple encoding for prediction (could be enhanced with training encodings)
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes

        return df

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has all required features in correct order"""

        # Add missing features with default values
        for feature in self.features:
            if feature not in df.columns:
                if 'flag' in feature or 'is_' in feature:
                    df[feature] = 0  # Boolean features default to 0
                else:
                    df[feature] = 0.0  # Numeric features default to 0.0

        # Select and order features to match training
        df = df[self.features]

        return df

    def predict(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make risk prediction on input features

        Args:
            input_features: Dictionary of transaction features

        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess features
            processed_features = self._preprocess_features(input_features)

            # Make prediction
            prediction_proba = self.model.predict_proba(processed_features)[0]
            prediction_class = self.model.predict(processed_features)[0]

            # Get class probabilities
            class_names = self.model.classes_
            class_proba_dict = dict(zip(class_names, prediction_proba))

            # Determine risk score and flag
            risk_score = float(prediction_proba.max())
            risk_flag = str(prediction_class)
            confidence = float(prediction_proba.max())

            # Map to standard risk categories if needed
            if risk_flag not in ['legit', 'suspicious', 'fraud']:
                risk_mapping = {0: 'legit', 1: 'suspicious', 2: 'fraud'}
                risk_flag = risk_mapping.get(int(risk_flag), risk_flag)

            result = {
                "risk_score": risk_score,
                "risk_flag": risk_flag,
                "confidence": confidence,
                "class_probabilities": class_proba_dict,
                "model_version": self.model_package.get('model_type', 'XGBoost'),
                "feature_count": len(self.features)
            }

            logger.info(f"Prediction completed: {risk_flag} (score: {risk_score:.4f})")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction error: {str(e)}")

    def validate_input_features(self, features: Dict[str, Any]) -> List[str]:
        """
        Validate input features and return list of issues

        Args:
            features: Input feature dictionary

        Returns:
            List of validation error messages
        """
        errors = []

        # Check required core features
        required_features = [
            'avg_tx_amount', 'device_change_freq', 'tx_hour', 'location_change_freq',
            'is_new_device', 'transaction_count_24h', 'time_since_last_tx',
            'tx_amount_to_balance_ratio', 'ip_address_reputation', 'is_weekend',
            'transaction_velocity_10min', 'country_change_flag'
        ]

        for feature in required_features:
            if feature not in features:
                errors.append(f"Missing required feature: {feature}")

        # Validate ranges
        if 'tx_hour' in features:
            if not 0 <= features['tx_hour'] <= 23:
                errors.append("tx_hour must be between 0 and 23")

        if 'device_change_freq' in features:
            if not 0 <= features['device_change_freq'] <= 1:
                errors.append("device_change_freq must be between 0 and 1")

        return errors