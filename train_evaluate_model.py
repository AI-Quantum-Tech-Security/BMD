import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import os
import json
import numpy as np
import warnings

# New imports for enhanced models
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

# Global Configuration
DATA_FILE = 'files/synthetic_behavioral_dataset.csv'
MODEL_OUTPUT_FILE = 'files/enhanced_model.pkl'
TARGET_COLUMN = 'risk_label'
EVAL_REPORT_FILE = 'files/enhanced_risk_model_eval.md'
MODEL_FEATURES_FILE = 'files/enhanced_model_features.json'

# Enhanced Feature Definitions - Updated to match your data
NUMERIC_FEATURES_FOR_MODEL = [
    'session_duration',
    'avg_tx_amount',
    'tx_amount',
    'device_change_freq',
    'location_change_freq',
    'account_age_days',
    'amount_balance_ratio',
    'typing_pattern_similarity',
    'mouse_movement_similarity',
    'distance_from_usual_location',
    'transaction_count_24h',
    'transaction_velocity_10min',
    'transaction_volume_24h',
    'time_since_last_tx',
    'merchant_risk_score',
    'ip_address_reputation',
    'login_frequency_7d',
    'failed_login_attempts',
    'mobility_score',
    'tx_intensity',
    'auth_risk_score'
]

BOOLEAN_FEATURES_FOR_MODEL = [
    'is_new_device',
    'is_weekend',
    'is_holiday',
    'is_business_hours',
    'is_unusual_hour',
    'country_change_flag',
    'vpn_proxy_flag',
    'is_vpn_detected',
    'is_high_risk_merchant'
]

CATEGORICAL_FEATURES_FOR_MODEL = [
    'merchant_id',
    'device_type',
    'location'
]

TIME_FEATURES_FOR_MODEL = [
    'timestamp',
    'tx_hour'
]


def add_enhanced_noise_to_features(X, noise_level=0.015):
    """
    Add sophisticated noise to numerical features to prevent perfect separation
    """
    X_noisy = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in X_noisy.columns and X_noisy[col].std() > 0:
            feature_std = X_noisy[col].std()
            adaptive_noise = np.random.normal(0, noise_level * feature_std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + adaptive_noise

            # Add occasional outliers (1% of data)
            outlier_mask = np.random.random(len(X_noisy)) < 0.01
            if outlier_mask.any():
                outlier_noise = np.random.normal(0, feature_std * 0.5, size=outlier_mask.sum())
                X_noisy.loc[outlier_mask, col] += outlier_noise

    return X_noisy


def create_advanced_feature_interactions(X):
    """
    Create sophisticated feature interactions for behavioral analysis
    """
    X_enhanced = X.copy()

    # Transaction amount interactions
    if 'tx_amount' in X.columns and 'avg_tx_amount' in X.columns:
        X_enhanced['tx_amount_ratio'] = X['tx_amount'] / (X['avg_tx_amount'] + 1e-8)
        X_enhanced['tx_amount_deviation'] = abs(X['tx_amount'] - X['avg_tx_amount'])

    # Velocity and behavioral interactions
    if 'session_duration' in X.columns and 'transaction_count_24h' in X.columns:
        X_enhanced['session_tx_intensity'] = X['transaction_count_24h'] / (X['session_duration'] + 1)

    # Risk behavior combinations
    if 'device_change_freq' in X.columns and 'location_change_freq' in X.columns:
        X_enhanced['behavior_volatility'] = X['device_change_freq'] + X['location_change_freq']
        X_enhanced['mobility_risk'] = X['device_change_freq'] * X['location_change_freq']

    # Temporal patterns
    if 'tx_hour' in X.columns:
        X_enhanced['is_business_hours_derived'] = ((X['tx_hour'] >= 9) & (X['tx_hour'] <= 17)).astype(int)
        X_enhanced['is_late_night_derived'] = ((X['tx_hour'] >= 23) | (X['tx_hour'] <= 5)).astype(int)

    # Network risk combinations
    if 'ip_address_reputation' in X.columns and 'vpn_proxy_flag' in X.columns:
        X_enhanced['network_risk_combined'] = X['ip_address_reputation'] * (1 + X['vpn_proxy_flag'])

    # Behavioral consistency score
    if 'typing_pattern_similarity' in X.columns and 'mouse_movement_similarity' in X.columns:
        X_enhanced['behavioral_consistency'] = (X['typing_pattern_similarity'] + X['mouse_movement_similarity']) / 2

    return X_enhanced


def calculate_enhanced_multiclass_roc_auc(y_true, y_pred_proba, classes):
    """
    Enhanced ROC-AUC calculation with better error handling
    """
    try:
        # Handle case where we have very few samples of some classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) < len(classes):
            print(f"Warning: Not all classes present in test set. Found: {unique_classes}")
            # Filter to only classes present in test set
            present_classes = [c for c in classes if c in unique_classes]
            if len(present_classes) < 2:
                return {}, None, None
            classes = present_classes

        y_true_binarized = label_binarize(y_true, classes=classes)

        if len(classes) == 2:
            y_true_binarized = np.column_stack([1 - y_true_binarized.ravel(), y_true_binarized.ravel()])

        roc_auc_per_class = {}
        valid_aucs = []

        for i, class_name in enumerate(classes):
            try:
                if len(classes) == 2 and i == 0:
                    continue

                # Check if we have both positive and negative samples
                y_true_class = y_true_binarized[:, i] if len(classes) > 2 else y_true_binarized[:, 1]
                y_pred_class = y_pred_proba[:, i]

                if len(np.unique(y_true_class)) > 1:
                    auc_score = roc_auc_score(y_true_class, y_pred_class)
                    roc_auc_per_class[class_name] = auc_score
                    valid_aucs.append(auc_score)
                else:
                    roc_auc_per_class[class_name] = None
                    print(f"Warning: Class {class_name} has only one unique value in test set")

            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC for class {class_name}: {e}")
                roc_auc_per_class[class_name] = None

        if valid_aucs:
            macro_auc = np.mean(valid_aucs)
            weighted_auc = macro_auc  # Simplified for now
            return roc_auc_per_class, macro_auc, weighted_auc
        else:
            return roc_auc_per_class, None, None

    except Exception as e:
        print(f"Error calculating enhanced multiclass ROC-AUC: {e}")
        return {}, None, None


def calculate_enhanced_per_class_metrics(y_true, y_pred, classes):
    """
    Enhanced per-class metrics calculation
    """
    per_class_metrics = {}

    for class_name in classes:
        y_true_binary = (y_true == class_name).astype(int)
        y_pred_binary = (y_pred == class_name).astype(int)

        try:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            support = np.sum(y_true_binary)
            predicted_positive = np.sum(y_pred_binary)

            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'predicted_positive': predicted_positive
            }
        except Exception as e:
            print(f"Warning: Could not calculate metrics for class {class_name}: {e}")
            per_class_metrics[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'support': 0,
                'predicted_positive': 0
            }

    return per_class_metrics


def compare_enhanced_models(X_train, X_test, y_train, y_test):
    """
    Enhanced model comparison with XGBoost, LightGBM and improved configurations
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1500,
            solver='liblinear',
            penalty='l2',
            C=0.1
        ),
        'SVM': SVC(
            random_state=42,
            class_weight='balanced',
            probability=True,
            kernel='rbf',
            C=0.5,
            gamma='scale'
        )
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        )

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            num_leaves=40,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            class_weight='balanced',
            verbose=-1,
            n_jobs=-1
        )

    results = {}

    # Enhanced data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        print(f"\nTraining enhanced model: {name}")

        try:
            # Use scaled data for models that benefit from it
            if name in ['LogisticRegression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                X_train_cv = X_train_scaled
            else:
                # Tree-based models work better with original features
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                X_train_cv = X_train

            # Enhanced cross-validation with stratified sampling
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True,
                                          random_state=42)  # Reduced to 3 due to small classes
            cv_scores = cross_val_score(model, X_train_cv, y_train,
                                        cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)

            # Calculate comprehensive metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Enhanced ROC-AUC calculation
            classes = model.classes_
            roc_auc_per_class, macro_auc, weighted_auc = calculate_enhanced_multiclass_roc_auc(
                y_test, y_pred_proba, classes
            )

            results[name] = {
                'model': model,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc_macro': macro_auc,
                'roc_auc_weighted': weighted_auc,
                'roc_auc_per_class': roc_auc_per_class,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'scaler': scaler if name in ['LogisticRegression', 'SVM'] else None
            }

            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC (macro): {macro_auc:.4f}" if macro_auc else "  ROC-AUC: Could not calculate")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        except Exception as e:
            print(f"Error training model {name}: {e}")
            continue

    return results


def train_evaluate_enhanced_model():
    """
    Enhanced model training and evaluation with advanced techniques
    """
    print("Starting Enhanced Behavioral Risk Model Training and Evaluation")

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run data generation first.")
        return

    # Load and validate data
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Data cleaning and validation
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    if df.empty:
        print(f"Error: No data remaining after removing missing target values.")
        return

    # Enhanced target distribution analysis
    print(f"\nTarget distribution analysis:")
    target_counts = df[TARGET_COLUMN].value_counts()
    target_proportions = df[TARGET_COLUMN].value_counts(normalize=True)

    for label in target_counts.index:
        count = target_counts[label]
        prop = target_proportions[label]
        print(f"  {label}: {count} samples ({prop:.3f})")

    # Feature preparation - use only features that exist in the data
    all_possible_features = (
            NUMERIC_FEATURES_FOR_MODEL +
            BOOLEAN_FEATURES_FOR_MODEL +
            CATEGORICAL_FEATURES_FOR_MODEL +
            TIME_FEATURES_FOR_MODEL
    )

    existing_features = [col for col in all_possible_features if col in df.columns]
    missing_features = set(all_possible_features) - set(existing_features)

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")

    print(f"Using {len(existing_features)} features for modeling")

    X = df[existing_features].copy()
    y = df[TARGET_COLUMN].copy()

    print("\nStarting enhanced preprocessing pipeline...")

    # Enhanced time feature processing
    for col in TIME_FEATURES_FOR_MODEL:
        if col in X.columns:
            if col == 'timestamp':
                X[col] = pd.to_datetime(X[col], errors='coerce')
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_day_of_week'] = X[col].dt.dayofweek
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                X = X.drop(columns=[col])
                print(f"Enhanced time processing for: {col}")
            elif col == 'tx_hour':
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce')

    # Enhanced categorical feature processing
    categorical_features_in_X = [col for col in CATEGORICAL_FEATURES_FOR_MODEL if col in X.columns]
    if categorical_features_in_X:
        print(f"Processing categorical features: {categorical_features_in_X}")

        for col in categorical_features_in_X:
            value_counts = X[col].value_counts()

            # Advanced cardinality reduction
            if len(value_counts) > 15:
                # Keep top 12 values, group rest as 'Other'
                top_values = value_counts.head(12).index.tolist()
                X[col] = X[col].apply(lambda x: x if x in top_values else 'Other')
                print(f"Reduced cardinality for {col}: {len(value_counts)} -> 13 categories")

        # Apply one-hot encoding
        X = pd.get_dummies(X, columns=categorical_features_in_X, drop_first=True)
        print(f"One-hot encoding completed. New shape: {X.shape}")

    # FIXED: Enhanced missing value handling
    print("Handling missing values...")
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['int64', 'float64']:
                # Use median for skewed distributions, mean for normal
                if X[col].std() > 0:
                    skewness = X[col].skew()
                    if abs(skewness) > 1:  # Highly skewed
                        fill_value = X[col].median()
                        method = "median"
                    else:
                        fill_value = X[col].mean()
                        method = "mean"
                else:
                    fill_value = 0
                    method = "zero"

                X[col].fillna(fill_value, inplace=True)
                print(f"Imputed {col} with {method}: {fill_value:.4f}")
            else:
                # For non-numeric columns, fill with mode or 'Unknown'
                if len(X[col].mode()) > 0:
                    fill_value = X[col].mode()[0]
                else:
                    fill_value = 'Unknown'
                X[col].fillna(fill_value, inplace=True)
                print(f"Imputed {col} with: {fill_value}")

    # Advanced feature engineering
    print("Creating advanced feature interactions...")
    X = create_advanced_feature_interactions(X)
    print(f"Feature engineering completed. Final shape: {X.shape}")

    # Enhanced noise addition for regularization
    print("Adding sophisticated noise for regularization...")
    X = add_enhanced_noise_to_features(X, noise_level=0.01)

    # Final validation
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"Error: Non-numeric columns remaining: {non_numeric_cols}")
        return

    print("Enhanced preprocessing completed successfully")

    # Save enhanced feature list
    print(f"Saving enhanced feature list to {MODEL_FEATURES_FILE}")
    os.makedirs(os.path.dirname(MODEL_FEATURES_FILE), exist_ok=True)

    feature_metadata = {
        'features': X.columns.tolist(),
        'feature_count': len(X.columns),
        'original_features': existing_features,
        'engineered_features': [col for col in X.columns if col not in existing_features],
        'preprocessing_date': pd.Timestamp.now().isoformat()
    }

    with open(MODEL_FEATURES_FILE, 'w') as f:
        json.dump(feature_metadata, f, indent=2)

    # Enhanced data splitting with stratification
    # Check if we have enough samples for stratification
    min_class_size = target_counts.min()
    test_size = 0.25 if min_class_size >= 4 else 1 / min_class_size  # Ensure at least 1 sample per class in test
    test_size = min(test_size, 0.3)  # Cap at 30%

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nData split completed:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    # Class imbalance analysis
    train_class_counts = y_train.value_counts()
    imbalance_ratio = train_class_counts.max() / train_class_counts.min()

    print(f"\nClass distribution in training set:")
    for label in train_class_counts.index:
        print(f"  {label}: {train_class_counts[label]} ({train_class_counts[label] / len(y_train):.3f})")

    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    # Apply SMOTE carefully for extreme imbalance
    if imbalance_ratio > 3.0 and min(train_class_counts) >= 2:
        print(f"\nApplying SMOTE for class balancing...")
        try:
            # Use smaller k_neighbors for small classes
            k_neighbors = min(5, min(train_class_counts) - 1)
            k_neighbors = max(1, k_neighbors)

            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"SMOTE completed. New training size: {X_train_balanced.shape[0]}")

            balanced_counts = pd.Series(y_train_balanced).value_counts()
            print("New class distribution:")
            for label in balanced_counts.index:
                print(f"  {label}: {balanced_counts[label]}")

            X_train, y_train = X_train_balanced, y_train_balanced
        except Exception as e:
            print(f"SMOTE failed: {e}. Continuing with original data.")
    else:
        print("Skipping SMOTE due to extremely small class sizes")

    # Enhanced model comparison
    print("\n" + "=" * 60)
    print("ENHANCED MODEL COMPARISON")
    print("=" * 60)

    model_results = compare_enhanced_models(X_train, X_test, y_train, y_test)

    if not model_results:
        print("Error: No models were successfully trained")
        return

    # Select best model using composite score
    def calculate_composite_score(results):
        """Calculate composite score for model selection"""
        f1 = results['f1']
        cv_mean = results['cv_mean']
        cv_stability = 1 - min(results['cv_std'] / (results['cv_mean'] + 1e-8), 1.0)  # Prevent division by zero
        roc_auc = results['roc_auc_macro'] if results['roc_auc_macro'] else 0.5

        # Weighted composite score
        return 0.4 * f1 + 0.3 * cv_mean + 0.2 * cv_stability + 0.1 * roc_auc

    best_model_name = max(model_results.keys(), key=lambda x: calculate_composite_score(model_results[x]))
    best_model_info = model_results[best_model_name]
    best_model = best_model_info['model']

    print(f"\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print("=" * 60)

    composite_score = calculate_composite_score(best_model_info)
    print(f"Composite Score: {composite_score:.4f}")
    print(f"F1-Score: {best_model_info['f1']:.4f}")
    print(f"ROC-AUC (macro): {best_model_info['roc_auc_macro']:.4f}" if best_model_info[
        'roc_auc_macro'] else "ROC-AUC: N/A")
    print(f"CV Score: {best_model_info['cv_mean']:.4f} (+/- {best_model_info['cv_std'] * 2:.4f})")

    # Detailed evaluation of best model
    y_pred = best_model_info['y_pred']
    y_pred_proba = best_model_info['y_pred_proba']
    classes = best_model.classes_

    # Enhanced confusion matrix analysis
    print(f"\n" + "=" * 40)
    print("DETAILED EVALUATION")
    print("=" * 40)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"Classes: {list(classes)}")
    print(conf_matrix)

    # Per-class detailed metrics
    print(f"\nPer-Class Performance Analysis:")
    per_class_metrics = calculate_enhanced_per_class_metrics(y_test, y_pred, classes)

    for class_name, metrics in per_class_metrics.items():
        print(f"\nClass '{class_name}':")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Support: {metrics['support']}")

    # Overall performance metrics
    overall_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nOverall Performance:")
    print(f"  Weighted Precision: {overall_precision:.4f}")
    print(f"  Weighted Recall: {overall_recall:.4f}")
    print(f"  Weighted F1-Score: {overall_f1:.4f}")

    # Feature importance analysis
    print(f"\nFeature Importance Analysis:")
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(
            best_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        print("Top 20 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importances.head(20).items(), 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")
    else:
        print("Feature importance not available for this model type")

    # Save enhanced model
    print(f"\nSaving enhanced model to: {MODEL_OUTPUT_FILE}")
    os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)

    try:
        enhanced_model_package = {
            'model': best_model,
            'model_type': best_model_name,
            'scaler': best_model_info['scaler'],
            'features': X.columns.tolist(),
            'feature_metadata': feature_metadata,
            'training_metrics': {
                'f1_score': best_model_info['f1'],
                'roc_auc_macro': best_model_info['roc_auc_macro'],
                'cv_mean': best_model_info['cv_mean'],
                'cv_std': best_model_info['cv_std'],
                'composite_score': composite_score
            },
            'classes': list(classes),
            'training_date': pd.Timestamp.now().isoformat()
        }

        joblib.dump(enhanced_model_package, MODEL_OUTPUT_FILE)
        print(f"Enhanced model package saved successfully")

    except Exception as e:
        print(f"Error saving model: {e}")

    print(f"\n" + "=" * 60)
    print("ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    train_evaluate_enhanced_model()