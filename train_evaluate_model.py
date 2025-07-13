import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Core ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, label_binarize
)
from imblearn.over_sampling import SMOTE
import joblib

# Enhanced model imports with graceful fallback
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')


class EnhancedBehavioralModelTrainer:
    """
    Aggregated and enhanced behavioral risk model trainer with advanced ML capabilities
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced model trainer"""
        self.config = self._load_config(config)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_metadata = {}
        self.results = {}

    def _load_config(self, config: Optional[Dict] = None) -> Dict:
        """Load and validate configuration"""
        default_config = {
            'data_file': 'files/synthetic_behavioral_dataset.csv',
            'model_output_file': 'files/enhanced_model.pkl',
            'target_column': 'risk_label',
            'eval_report_file': 'files/enhanced_risk_model_eval.md',
            'model_features_file': 'files/enhanced_model_features.json',
            'test_size': 0.25,
            'cv_folds': 3,
            'random_state': 42,
            'noise_level': 0.01,
            'smote_threshold': 3.0
        }

        if config:
            default_config.update(config)

        return default_config

    def _get_feature_lists(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get comprehensive feature lists for the model"""
        numeric_features = [
            'session_duration', 'avg_tx_amount', 'tx_amount', 'device_change_freq',
            'location_change_freq', 'account_age_days', 'amount_balance_ratio',
            'typing_pattern_similarity', 'mouse_movement_similarity',
            'distance_from_usual_location', 'transaction_count_24h',
            'transaction_velocity_10min', 'transaction_volume_24h',
            'time_since_last_tx', 'merchant_risk_score', 'ip_address_reputation',
            'login_frequency_7d', 'failed_login_attempts', 'mobility_score',
            'tx_intensity', 'auth_risk_score'
        ]

        boolean_features = [
            'is_new_device', 'is_weekend', 'is_holiday', 'is_business_hours',
            'is_unusual_hour', 'country_change_flag', 'vpn_proxy_flag',
            'is_vpn_detected', 'is_high_risk_merchant'
        ]

        categorical_features = ['merchant_id', 'device_type', 'location']
        time_features = ['timestamp', 'tx_hour']

        return numeric_features, boolean_features, categorical_features, time_features

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate input data"""
        if not os.path.exists(self.config['data_file']):
            raise FileNotFoundError(f"Data file '{self.config['data_file']}' not found")

        df = pd.read_csv(self.config['data_file'])
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        target_col = self.config['target_column']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Clean target data
        df.dropna(subset=[target_col], inplace=True)
        if df.empty:
            raise ValueError("No data remaining after cleaning")

        # Analyze target distribution
        target_counts = df[target_col].value_counts()
        print("\nTarget distribution:")
        for label, count in target_counts.items():
            prop = count / len(df)
            print(f"  {label}: {count} samples ({prop:.3f})")

        return df, df[target_col]

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive feature preprocessing pipeline"""
        numeric_features, boolean_features, categorical_features, time_features = self._get_feature_lists()

        # Select existing features
        all_features = numeric_features + boolean_features + categorical_features + time_features
        existing_features = [col for col in all_features if col in df.columns]
        missing_features = set(all_features) - set(existing_features)

        if missing_features:
            print(f"Warning: Missing features: {missing_features}")

        print(f"Using {len(existing_features)} features for modeling")
        X = df[existing_features].copy()

        # Process time features
        X = self._process_time_features(X, time_features)

        # Process categorical features
        X = self._process_categorical_features(X, categorical_features)

        # Handle missing values
        X = self._handle_missing_values(X)

        # Create feature interactions
        X = self._create_feature_interactions(X)

        # Add regularization noise
        X = self._add_regularization_noise(X)

        # Store feature metadata
        self.feature_metadata = {
            'features': X.columns.tolist(),
            'feature_count': len(X.columns),
            'original_features': existing_features,
            'engineered_features': [col for col in X.columns if col not in existing_features],
            'preprocessing_date': datetime.now().isoformat()
        }

        print(f"Preprocessing completed. Final shape: {X.shape}")
        return X

    def _process_time_features(self, X: pd.DataFrame, time_features: List[str]) -> pd.DataFrame:
        """Process temporal features"""
        for col in time_features:
            if col not in X.columns:
                continue

            if col == 'timestamp':
                X[col] = pd.to_datetime(X[col], errors='coerce')
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_day_of_week'] = X[col].dt.dayofweek
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                X = X.drop(columns=[col])
                print(f"Processed time feature: {col}")
            elif col == 'tx_hour' and X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')

        return X

    def _process_categorical_features(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """Process categorical features with cardinality reduction"""
        categorical_in_X = [col for col in categorical_features if col in X.columns]

        if not categorical_in_X:
            return X

        print(f"Processing categorical features: {categorical_in_X}")

        for col in categorical_in_X:
            value_counts = X[col].value_counts()
            if len(value_counts) > 15:
                top_values = value_counts.head(12).index.tolist()
                X[col] = X[col].apply(lambda x: x if x in top_values else 'Other')
                print(f"Reduced cardinality for {col}: {len(value_counts)} -> 13 categories")

        X = pd.get_dummies(X, columns=categorical_in_X, drop_first=True)
        print(f"One-hot encoding completed. Shape: {X.shape}")

        return X

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        for col in X.columns:
            if not X[col].isnull().any():
                continue

            if X[col].dtype in ['int64', 'float64']:
                if X[col].std() > 0:
                    skewness = X[col].skew()
                    fill_value = X[col].median() if abs(skewness) > 1 else X[col].mean()
                    method = "median" if abs(skewness) > 1 else "mean"
                else:
                    fill_value = 0
                    method = "zero"
                X[col].fillna(fill_value, inplace=True)
                print(f"Imputed {col} with {method}: {fill_value:.4f}")
            else:
                fill_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                X[col].fillna(fill_value, inplace=True)
                print(f"Imputed {col} with: {fill_value}")

        return X

    def _create_feature_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced feature interactions"""
        X_enhanced = X.copy()

        # Transaction amount interactions
        if 'tx_amount' in X.columns and 'avg_tx_amount' in X.columns:
            X_enhanced['tx_amount_ratio'] = X['tx_amount'] / (X['avg_tx_amount'] + 1e-8)
            X_enhanced['tx_amount_deviation'] = abs(X['tx_amount'] - X['avg_tx_amount'])

        # Behavioral interactions
        if 'session_duration' in X.columns and 'transaction_count_24h' in X.columns:
            X_enhanced['session_tx_intensity'] = X['transaction_count_24h'] / (X['session_duration'] + 1)

        # Risk combinations
        if 'device_change_freq' in X.columns and 'location_change_freq' in X.columns:
            X_enhanced['behavior_volatility'] = X['device_change_freq'] + X['location_change_freq']
            X_enhanced['mobility_risk'] = X['device_change_freq'] * X['location_change_freq']

        # Network risk
        if 'ip_address_reputation' in X.columns and 'vpn_proxy_flag' in X.columns:
            X_enhanced['network_risk_combined'] = X['ip_address_reputation'] * (1 + X['vpn_proxy_flag'])

        # Behavioral consistency
        if 'typing_pattern_similarity' in X.columns and 'mouse_movement_similarity' in X.columns:
            X_enhanced['behavioral_consistency'] = (
                                                           X['typing_pattern_similarity'] + X[
                                                       'mouse_movement_similarity']
                                                   ) / 2

        print(f"Feature engineering completed: {X.shape[1]} -> {X_enhanced.shape[1]} features")
        return X_enhanced

    def _add_regularization_noise(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add sophisticated noise for regularization"""
        X_noisy = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if X_noisy[col].std() > 0:
                noise_std = self.config['noise_level'] * X_noisy[col].std()
                noise = np.random.normal(0, noise_std, size=len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise

                # Add occasional outliers
                outlier_mask = np.random.random(len(X_noisy)) < 0.01
                if outlier_mask.any():
                    outlier_noise = np.random.normal(0, X_noisy[col].std() * 0.5, size=outlier_mask.sum())
                    X_noisy.loc[outlier_mask, col] += outlier_noise

        return X_noisy

    def prepare_labels(self, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare labels for both string and numeric model requirements"""
        # Original string labels
        y_str = y.values

        # Encoded numeric labels for XGBoost/LightGBM
        y_encoded = self.label_encoder.fit_transform(y)

        return y_str, y_encoded

    def split_data(self, X: pd.DataFrame, y_str: np.ndarray, y_encoded: np.ndarray) -> Dict:
        """Split data with proper stratification"""
        # Calculate adaptive test size
        target_counts = pd.Series(y_str).value_counts()
        min_class_size = target_counts.min()
        test_size = min(self.config['test_size'], 0.3)

        if min_class_size < 4:
            test_size = max(1 / min_class_size, 0.1)

        # Split data
        split_data = train_test_split(
            X, y_str, y_encoded,
            test_size=test_size,
            random_state=self.config['random_state'],
            stratify=y_str
        )

        X_train, X_test, y_train_str, y_test_str, y_train_enc, y_test_enc = split_data

        print(f"\nData split completed:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")

        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train_str': y_train_str, 'y_test_str': y_test_str,
            'y_train_enc': y_train_enc, 'y_test_enc': y_test_enc
        }

    def apply_class_balancing(self, data_split: Dict) -> Dict:
        """Apply SMOTE for class balancing if needed"""
        X_train = data_split['X_train']
        y_train_str = data_split['y_train_str']
        y_train_enc = data_split['y_train_enc']

        # Analyze class imbalance
        train_counts = pd.Series(y_train_str).value_counts()
        imbalance_ratio = train_counts.max() / train_counts.min()

        print(f"\nClass distribution in training:")
        for label, count in train_counts.items():
            print(f"  {label}: {count} ({count / len(y_train_str):.3f})")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")

        # Apply SMOTE if needed
        if imbalance_ratio > self.config['smote_threshold'] and min(train_counts) >= 2:
            print("\nApplying SMOTE for class balancing...")
            try:
                k_neighbors = min(5, min(train_counts) - 1)
                k_neighbors = max(1, k_neighbors)

                smote = SMOTE(random_state=self.config['random_state'], k_neighbors=k_neighbors)
                X_train_balanced, y_train_str_balanced = smote.fit_resample(X_train, y_train_str)
                _, y_train_enc_balanced = smote.fit_resample(X_train, y_train_enc)

                balanced_counts = pd.Series(y_train_str_balanced).value_counts()
                print(f"SMOTE completed. New size: {X_train_balanced.shape[0]}")
                print("New distribution:")
                for label, count in balanced_counts.items():
                    print(f"  {label}: {count}")

                data_split.update({
                    'X_train': X_train_balanced,
                    'y_train_str': y_train_str_balanced,
                    'y_train_enc': y_train_enc_balanced
                })

            except Exception as e:
                print(f"SMOTE failed: {e}. Using original data.")
        else:
            print("Skipping SMOTE - insufficient data or balanced classes")

        return data_split

    def create_model_configs(self) -> Dict:
        """Create enhanced model configurations"""
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100, max_depth=12, min_samples_split=15,
                    min_samples_leaf=8, max_features='sqrt', random_state=self.config['random_state'],
                    class_weight='balanced', n_jobs=-1
                ),
                'use_encoded_labels': False,
                'scale_features': False
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=self.config['random_state'], class_weight='balanced',
                    max_iter=1500, solver='liblinear', penalty='l2', C=0.1
                ),
                'use_encoded_labels': False,
                'scale_features': True
            },
            'SVM': {
                'model': SVC(
                    random_state=self.config['random_state'], class_weight='balanced',
                    probability=True, kernel='rbf', C=0.5, gamma='scale'
                ),
                'use_encoded_labels': False,
                'scale_features': True
            }
        }

        # Add XGBoost with proper label handling
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': XGBClassifier(
                    n_estimators=150, max_depth=7, learning_rate=0.08,
                    subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=self.config['random_state'],
                    eval_metric='mlogloss', n_jobs=-1, verbosity=0
                ),
                'use_encoded_labels': True,
                'scale_features': False
            }

        # Add LightGBM with proper label handling
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': LGBMClassifier(
                    n_estimators=150, max_depth=7, learning_rate=0.08,
                    num_leaves=40, subsample=0.85, colsample_bytree=0.85,
                    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=self.config['random_state'], class_weight='balanced',
                    verbose=-1, n_jobs=-1
                ),
                'use_encoded_labels': True,
                'scale_features': False
            }

        return models

    def train_and_evaluate_models(self, data_split: Dict) -> Dict:
        """Train and evaluate all models"""
        models = self.create_model_configs()
        results = {}

        # Prepare scaled features
        X_train_scaled = self.scaler.fit_transform(data_split['X_train'])
        X_test_scaled = self.scaler.transform(data_split['X_test'])

        print("\n" + "=" * 60)
        print("ENHANCED MODEL COMPARISON")
        print("=" * 60)

        for name, config in models.items():
            print(f"\nTraining model: {name}")

            try:
                model = config['model']
                use_encoded = config['use_encoded_labels']
                scale_features = config['scale_features']

                # Select appropriate data
                X_train = X_train_scaled if scale_features else data_split['X_train']
                X_test = X_test_scaled if scale_features else data_split['X_test']
                y_train = data_split['y_train_enc'] if use_encoded else data_split['y_train_str']
                y_test = data_split['y_test_enc'] if use_encoded else data_split['y_test_str']

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Convert back to string labels if needed for evaluation
                if use_encoded:
                    y_test_eval = data_split['y_test_str']
                    y_pred_eval = self.label_encoder.inverse_transform(y_pred)
                else:
                    y_test_eval = y_test
                    y_pred_eval = y_pred

                # Cross-validation
                cv_strategy = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True,
                                              random_state=self.config['random_state'])
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy,
                                            scoring='f1_weighted', n_jobs=-1)

                # Calculate metrics
                precision = precision_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
                recall = recall_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
                f1 = f1_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)

                # ROC-AUC calculation
                roc_auc_macro = self._calculate_roc_auc(y_test_eval, y_pred_proba, model.classes_)

                results[name] = {
                    'model': model,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc_macro': roc_auc_macro,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred_eval,
                    'y_pred_proba': y_pred_proba,
                    'scaler': self.scaler if scale_features else None,
                    'use_encoded_labels': use_encoded
                }

                print(f"  F1-Score: {f1:.4f}")
                print(f"  ROC-AUC: {roc_auc_macro:.4f}" if roc_auc_macro else "  ROC-AUC: N/A")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        self.results = results
        return results

    def _calculate_roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           classes: np.ndarray) -> Optional[float]:
        """Calculate ROC-AUC with proper error handling"""
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                return None

            y_true_binarized = label_binarize(y_true, classes=classes)
            if len(classes) == 2:
                y_true_binarized = np.column_stack([1 - y_true_binarized.ravel(), y_true_binarized.ravel()])

            valid_aucs = []
            for i, class_name in enumerate(classes):
                try:
                    y_true_class = y_true_binarized[:, i] if len(classes) > 2 else y_true_binarized[:, 1]
                    y_pred_class = y_pred_proba[:, i]

                    if len(np.unique(y_true_class)) > 1:
                        auc = roc_auc_score(y_true_class, y_pred_class)
                        valid_aucs.append(auc)
                except:
                    continue

            return np.mean(valid_aucs) if valid_aucs else None

        except Exception as e:
            print(f"ROC-AUC calculation error: {e}")
            return None

    def select_best_model(self) -> Tuple[str, Dict]:
        """Select best model using composite scoring"""

        def composite_score(results):
            f1 = results['f1']
            cv_mean = results['cv_mean']
            cv_stability = 1 - min(results['cv_std'] / (results['cv_mean'] + 1e-8), 1.0)
            roc_auc = results['roc_auc_macro'] if results['roc_auc_macro'] else 0.5
            return 0.4 * f1 + 0.3 * cv_mean + 0.2 * cv_stability + 0.1 * roc_auc

        best_name = max(self.results.keys(), key=lambda x: composite_score(self.results[x]))
        best_results = self.results[best_name]

        print(f"\n" + "=" * 60)
        print(f"BEST MODEL: {best_name}")
        print("=" * 60)
        print(f"Composite Score: {composite_score(best_results):.4f}")
        print(f"F1-Score: {best_results['f1']:.4f}")
        print(f"ROC-AUC: {best_results['roc_auc_macro']:.4f}" if best_results['roc_auc_macro'] else "ROC-AUC: N/A")
        print(f"CV Score: {best_results['cv_mean']:.4f} (+/- {best_results['cv_std'] * 2:.4f})")

        return best_name, best_results

    def save_model_and_results(self, best_name: str, best_results: Dict,
                               data_split: Dict) -> None:
        """Save the best model and generate comprehensive report"""
        # Create directories
        os.makedirs(os.path.dirname(self.config['model_output_file']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['model_features_file']), exist_ok=True)

        # Save feature metadata
        with open(self.config['model_features_file'], 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)

        # Prepare model package
        model_package = {
            'model': best_results['model'],
            'model_type': best_name,
            'scaler': best_results['scaler'],
            'label_encoder': self.label_encoder,
            'features': self.feature_metadata['features'],
            'feature_metadata': self.feature_metadata,
            'training_metrics': {
                'f1_score': best_results['f1'],
                'roc_auc_macro': best_results['roc_auc_macro'],
                'cv_mean': best_results['cv_mean'],
                'cv_std': best_results['cv_std']
            },
            'classes': list(best_results['model'].classes_),
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }

        # Save model
        joblib.dump(model_package, self.config['model_output_file'])
        print(f"\nModel saved to: {self.config['model_output_file']}")

        # Generate detailed evaluation
        self._generate_evaluation_report(best_name, best_results, data_split)

    def _generate_evaluation_report(self, best_name: str, best_results: Dict,
                                    data_split: Dict) -> None:
        """Generate comprehensive evaluation report"""
        y_test = data_split['y_test_str']
        y_pred = best_results['y_pred']

        # Calculate confusion matrix and metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        classes = best_results['model'].classes_

        print(f"\nDetailed Evaluation:")
        print(f"Confusion Matrix:")
        print(f"Classes: {list(classes)}")
        print(conf_matrix)

        # Per-class metrics
        print(f"\nPer-Class Performance:")
        for class_name in classes:
            y_true_binary = (y_test == class_name).astype(int)
            y_pred_binary = (y_pred == class_name).astype(int)

            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            support = np.sum(y_true_binary)

            print(f"  {class_name}:")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1-Score: {f1:.4f}")
            print(f"    Support: {support}")

        # Feature importance
        if hasattr(best_results['model'], 'feature_importances_'):
            feature_names = self.feature_metadata['features']
            importances = pd.Series(
                best_results['model'].feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)

            print(f"\nTop 15 Feature Importances:")
            for i, (feature, importance) in enumerate(importances.head(15).items(), 1):
                print(f"  {i:2d}. {feature}: {importance:.4f}")

    def run_complete_pipeline(self) -> Dict:
        """Run the complete model training pipeline"""
        print("=" * 80)
        print("ENHANCED BEHAVIORAL RISK MODEL TRAINING PIPELINE")
        print("=" * 80)

        # Load and validate data
        df, y = self.load_and_validate_data()

        # Preprocess features
        X = self.preprocess_features(df)

        # Prepare labels
        y_str, y_encoded = self.prepare_labels(y)

        # Split data
        data_split = self.split_data(X, y_str, y_encoded)

        # Apply class balancing
        data_split = self.apply_class_balancing(data_split)

        # Train and evaluate models
        results = self.train_and_evaluate_models(data_split)

        # Select best model
        best_name, best_results = self.select_best_model()

        # Save model and generate report
        self.save_model_and_results(best_name, best_results, data_split)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return {
            'best_model': best_name,
            'results': results,
            'feature_metadata': self.feature_metadata
        }


def main():
    """Main execution function"""
    # Custom configuration (optional)
    config = {
        'cv_folds': 3,  # Reduced for small datasets
        'smote_threshold': 3.0,  # Apply SMOTE when imbalance > 3:1
        'noise_level': 0.01  # Light regularization noise
    }

    # Initialize and run trainer
    trainer = EnhancedBehavioralModelTrainer(config)
    results = trainer.run_complete_pipeline()

    print(f"\nFinal Results:")
    print(f"Best Model: {results['best_model']}")
    print(f"Total Features: {results['feature_metadata']['feature_count']}")
    print(f"Models Trained: {list(results['results'].keys())}")


if __name__ == "__main__":
    main()