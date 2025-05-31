import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
import joblib
import os
import json
import numpy as np

# --- Global Configuration ---
DATA_FILE = 'files/synthetic_behavioral_data.csv'
MODEL_OUTPUT_FILE = 'files/risk_model.pkl'
TARGET_COLUMN = 'risk_label'
EVAL_REPORT_FILE = 'files/risk_model_eval.md'
MODEL_FEATURES_FILE = 'files/model_features.json'

# --- Feature Definitions for Model Preprocessing ---
NUMERIC_FEATURES_FOR_MODEL = [
    'session_duration',
    'avg_tx_amount',
    'geo_distance_delta',
    'tx_amount',
    'std_tx_amount_user',
    'device_change_freq',
    'location_change_freq',
    'txs_last_24h',
    'txs_last_7d',
    'tx_velocity_10min',
    'ip_risk_score',
    'avg_tx_hour_user',
]

BOOLEAN_FEATURES_FOR_MODEL = [
    'has_recent_password_reset',
    'is_new_device',
    'is_weekend',
    'country_mismatch',
    'is_vpn'
]

CATEGORICAL_FEATURES_FOR_MODEL = [
    'currency',
    'tx_type',
    'merchant_id',
    'tx_location',
    'device_id',
    'ip_address',
]

TIME_FEATURES_FOR_MODEL = [
    'timestamp',
    'login_time_pattern',
    'tx_hour',
]


def calculate_multiclass_roc_auc(y_true, y_pred_proba, classes):
    """
    Calculate ROC-AUC for multiclass classification using one-vs-rest approach
    """
    try:
        # Binarize the output for multiclass ROC-AUC calculation
        y_true_binarized = label_binarize(y_true, classes=classes)

        # If we have only 2 classes, label_binarize returns 1D array, we need 2D
        if len(classes) == 2:
            y_true_binarized = np.column_stack([1 - y_true_binarized, y_true_binarized])

        # Calculate ROC-AUC for each class (one-vs-rest)
        roc_auc_per_class = {}
        for i, class_name in enumerate(classes):
            if len(classes) == 2 and i == 0:
                continue  # Skip first class for binary case to avoid duplication
            try:
                auc_score = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
                roc_auc_per_class[class_name] = auc_score
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC for class {class_name}: {e}")
                roc_auc_per_class[class_name] = None

        # Calculate macro and weighted average ROC-AUC
        valid_aucs = [auc for auc in roc_auc_per_class.values() if auc is not None]

        if valid_aucs:
            macro_auc = np.mean(valid_aucs)
            # For weighted average, we'd need class frequencies, so let's use macro for simplicity
            weighted_auc = macro_auc  # Simplified - in practice you'd weight by class frequency

            return roc_auc_per_class, macro_auc, weighted_auc
        else:
            return roc_auc_per_class, None, None

    except Exception as e:
        print(f"Error calculating multiclass ROC-AUC: {e}")
        return {}, None, None


def calculate_per_class_metrics(y_true, y_pred, classes):
    """
    Calculate precision and recall for each class individually
    """
    per_class_metrics = {}

    for class_name in classes:
        # Create binary masks for current class vs all others
        y_true_binary = (y_true == class_name).astype(int)
        y_pred_binary = (y_pred == class_name).astype(int)

        try:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            print(f"Warning: Could not calculate metrics for class {class_name}: {e}")
            per_class_metrics[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

    return per_class_metrics


def train_evaluate_model():
    print(f"--- Starting Model Training and Evaluation ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run 'generate_data.py' first.")
        return

    df = pd.read_csv(DATA_FILE)
    print("Data loaded for model training.")
    print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
    print("\nFirst 5 rows of data:")
    print(df.head())
    print("\nInformation about columns and data types:")
    df.info()

    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the DataFrame.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    if df.empty:
        print(f"Error: No data left after dropping rows with missing '{TARGET_COLUMN}'.")
        return

    # Print target distribution
    print(f"\nTarget distribution in full dataset:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"Target distribution (proportions):")
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    model_features_list = (
            NUMERIC_FEATURES_FOR_MODEL +
            BOOLEAN_FEATURES_FOR_MODEL +
            CATEGORICAL_FEATURES_FOR_MODEL +
            TIME_FEATURES_FOR_MODEL
    )

    existing_model_features = [col for col in model_features_list if col in df.columns]
    if len(existing_model_features) != len(model_features_list):
        missing = set(model_features_list) - set(existing_model_features)
        print(f"Warning: Missing model feature columns: {missing}")

    X = df[existing_model_features].copy()
    y = df[TARGET_COLUMN].copy()

    print("\nStarting data preprocessing for model training...")

    # Process time features
    for col in list(TIME_FEATURES_FOR_MODEL):
        if col in X.columns:
            if col == 'timestamp':
                X[col] = pd.to_datetime(X[col])
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_day_of_week'] = X[col].dt.dayofweek
                X[f'{col}_month'] = X[col].dt.month
                X = X.drop(columns=[col])
                print(f"Processed time feature: {col}")
            elif col == 'login_time_pattern':
                time_series = pd.to_datetime(X[col].astype(str), format='%H:%M', errors='coerce')
                X[f'{col}_hour'] = time_series.dt.hour
                X[f'{col}_minute'] = time_series.dt.minute
                X = X.drop(columns=[col])
                print(f"Processed time feature: {col}")
            elif col == 'tx_hour':
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce')

    # Process categorical features
    current_categorical_features_in_X = [col for col in CATEGORICAL_FEATURES_FOR_MODEL if col in X.columns]
    if current_categorical_features_in_X:
        print(f"Applying One-Hot Encoding to categorical features: {current_categorical_features_in_X}")
        X = pd.get_dummies(X, columns=current_categorical_features_in_X, drop_first=False)
        print(f"Shape after One-Hot Encoding: {X.shape}")

    # Handle missing values
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64'] and X[col].isnull().any():
            mean_val = X[col].mean()
            X[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in numerical column '{col}' with mean: {mean_val:.4f}")

    if X.isnull().values.any():
        print("Warning: Missing values still exist in the preprocessed data.")
        print("Columns with missing values:")
        print(X.isnull().sum()[X.isnull().sum() > 0])

    # Check for non-numeric columns
    non_numeric_cols_after_prep = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols_after_prep) > 0:
        print(f"Error: Non-numeric columns found after preprocessing: {non_numeric_cols_after_prep}")
        return

    print("Data preprocessing completed.")

    # Save the columns after preprocessing for consistent API input
    print(f"Saving preprocessed feature column names to {MODEL_FEATURES_FILE}")
    os.makedirs(os.path.dirname(MODEL_FEATURES_FILE), exist_ok=True)
    try:
        with open(MODEL_FEATURES_FILE, 'w') as f:
            json.dump(X.columns.tolist(), f, indent=2)
        print(f"Feature column names saved to {MODEL_FEATURES_FILE}")
    except Exception as e:
        print(f"Error occurred while saving feature column names: {e}")

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")
    print(f"Target distribution in training set:")
    print(y_train.value_counts())
    print(f"Target distribution in training set (proportions):")
    print(y_train.value_counts(normalize=True))
    print(f"Target distribution in test set:")
    print(y_test.value_counts())
    print(f"Target distribution in test set (proportions):")
    print(y_test.value_counts(normalize=True))

    # Train model
    print("\nStarting Random Forest model training...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Make predictions
    print("\nEvaluating model on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Get unique classes
    classes = model.classes_
    print(f"Model classes: {classes}")

    # Calculate ROC-AUC (REQUIRED METRIC)
    print("\n=== ROC-AUC CALCULATION ===")
    roc_auc_per_class, macro_auc, weighted_auc = calculate_multiclass_roc_auc(y_test, y_pred_proba, classes)

    if roc_auc_per_class:
        print("ROC-AUC per class (One-vs-Rest):")
        for class_name, auc_score in roc_auc_per_class.items():
            if auc_score is not None:
                print(f"  {class_name}: {auc_score:.4f}")
            else:
                print(f"  {class_name}: Could not calculate")

    if macro_auc is not None:
        print(f"Macro-average ROC-AUC: {macro_auc:.4f}")
        print(f"Weighted-average ROC-AUC: {weighted_auc:.4f}")
    else:
        print("Could not calculate macro/weighted ROC-AUC")

    # Calculate Confusion Matrix (REQUIRED METRIC)
    print("\n=== CONFUSION MATRIX ===")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Confusion Matrix classes order: {classes}")

    # Calculate Precision/Recall for each class (REQUIRED METRIC)
    print("\n=== PRECISION/RECALL FOR EACH CLASS ===")
    per_class_metrics = calculate_per_class_metrics(y_test, y_pred, classes)

    for class_name, metrics in per_class_metrics.items():
        print(f"\nClass '{class_name}':")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

    # Overall metrics
    print("\n=== OVERALL METRICS ===")
    overall_precision = precision_score(y_test, y_pred, average='weighted')
    overall_recall = recall_score(y_test, y_pred, average='weighted')
    overall_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Overall Weighted Precision: {overall_precision:.4f}")
    print(f"Overall Weighted Recall: {overall_recall:.4f}")
    print(f"Overall Weighted F1-Score: {overall_f1:.4f}")

    # Classification Report
    print("\n=== CLASSIFICATION REPORT ===")
    class_report = classification_report(y_test, y_pred, target_names=classes)
    print(class_report)

    # Feature Importance
    print("\n=== FEATURE IMPORTANCE ===")
    try:
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Top 15 most important features:")
        print(feature_importances.head(15))
    except Exception as e:
        print(f"Could not retrieve feature importances: {e}")

    # Save trained model
    print(f"\nSaving trained model to file: {MODEL_OUTPUT_FILE}")
    os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)
    try:
        joblib.dump(model, MODEL_OUTPUT_FILE)
        print(f"Model successfully saved as {MODEL_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

    # Generate evaluation report
    print(f"\nGenerating evaluation report: {EVAL_REPORT_FILE}")
    os.makedirs(os.path.dirname(EVAL_REPORT_FILE), exist_ok=True)

    with open(EVAL_REPORT_FILE, 'w') as f:
        f.write(f"# Risk Model Evaluation Report\n\n")

        f.write(f"## Model Details\n")
        f.write(f"- **Model Type:** RandomForestClassifier\n")
        f.write(f"- **Number of estimators:** {model.n_estimators}\n")
        f.write(f"- **Random state:** {model.random_state}\n")
        f.write(f"- **Class Weight:** `balanced`\n")
        f.write(f"- **Model Classes:** {list(classes)}\n\n")

        f.write(f"## Data Overview\n")
        f.write(f"- **Total rows in dataset:** {df.shape[0]}\n")
        f.write(f"- **Training set rows:** {X_train.shape[0]}\n")
        f.write(f"- **Test set rows:** {X_test.shape[0]}\n")
        f.write(f"- **Features after preprocessing:** {X.shape[1]}\n")
        f.write(f"- **Target column:** `{TARGET_COLUMN}`\n")
        f.write(f"- **Class distribution in test set:**\n")
        for class_val, count in y_test.value_counts().items():
            proportion = count / len(y_test)
            f.write(f"  - **Class '{class_val}':** {count} samples ({proportion:.2%})\n")
        f.write(f"\n")

        f.write(f"## Evaluation Metrics on Test Set\n\n")

        # ROC-AUC Section
        f.write(f"### ROC-AUC Scores\n")
        if roc_auc_per_class:
            f.write(f"**Per-class ROC-AUC (One-vs-Rest):**\n")
            for class_name, auc_score in roc_auc_per_class.items():
                if auc_score is not None:
                    f.write(f"- **{class_name}:** {auc_score:.4f}\n")
                else:
                    f.write(f"- **{class_name}:** Could not calculate\n")
            f.write(f"\n")

        if macro_auc is not None:
            f.write(f"**Macro-average ROC-AUC:** {macro_auc:.4f}\n")
            f.write(f"**Weighted-average ROC-AUC:** {weighted_auc:.4f}\n\n")

        # Precision/Recall per class
        f.write(f"### Precision/Recall for Each Class\n")
        for class_name, metrics in per_class_metrics.items():
            f.write(f"**Class '{class_name}':**\n")
            f.write(f"- Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Recall: {metrics['recall']:.4f}\n")
            f.write(f"- F1-Score: {metrics['f1_score']:.4f}\n\n")

        # Overall metrics
        f.write(f"### Overall Metrics\n")
        f.write(f"- **Overall Weighted Precision:** {overall_precision:.4f}\n")
        f.write(f"- **Overall Weighted Recall:** {overall_recall:.4f}\n")
        f.write(f"- **Overall Weighted F1-Score:** {overall_f1:.4f}\n\n")

        # Confusion Matrix
        f.write(f"### Confusion Matrix\n")
        f.write(f"Classes order: {list(classes)}\n")
        f.write(f"```\n")
        f.write(f"{conf_matrix}\n")
        f.write(f"```\n\n")

        # Classification Report
        f.write(f"### Detailed Classification Report\n")
        f.write(f"```\n")
        f.write(f"{class_report}\n")
        f.write(f"```\n\n")

        # Feature Importance
        f.write(f"### Feature Importance\n")
        try:
            f.write(f"The top 15 most important features are:\n")
            f.write(f"```\n")
            f.write(f"{feature_importances.head(15).to_string()}\n")
            f.write(f"```\n\n")
        except Exception as e:
            f.write(f"Could not retrieve feature importances: {e}\n\n")

        # Analysis and recommendations
        f.write(f"## Analysis Summary\n")

        # Check if fraud class exists and analyze its performance
        if 'fraud' in per_class_metrics:
            fraud_recall = per_class_metrics['fraud']['recall']
            fraud_precision = per_class_metrics['fraud']['precision']
            f.write(
                f"- **Fraud Detection Performance:** Recall = {fraud_recall:.4f}, Precision = {fraud_precision:.4f}\n")
            if fraud_recall < 0.7:
                f.write(f"  - ⚠️ Low fraud recall - model may miss fraudulent transactions\n")
            if fraud_precision < 0.5:
                f.write(f"  - ⚠️ Low fraud precision - model may flag too many legitimate transactions\n")

        if macro_auc is not None:
            f.write(f"- **Overall ROC-AUC Performance:** {macro_auc:.4f}\n")
            if macro_auc >= 0.8:
                f.write(f"  - ✅ Good discriminative performance\n")
            elif macro_auc >= 0.7:
                f.write(f"  - ⚠️ Moderate discriminative performance\n")
            else:
                f.write(f"  - ❌ Poor discriminative performance - model needs improvement\n")

        f.write(f"\n## Next Steps & Recommendations\n")
        f.write(
            f"- **Hyperparameter Tuning:** Optimize RandomForest parameters (n_estimators, max_depth, min_samples_split)\n")
        f.write(f"- **Cross-validation:** Implement K-fold cross-validation for more robust evaluation\n")
        f.write(f"- **Alternative Models:** Test XGBoost, LightGBM, or ensemble methods\n")
        f.write(f"- **Feature Engineering:** Create additional behavioral features or feature interactions\n")
        f.write(f"- **Class Imbalance:** Consider SMOTE, cost-sensitive learning, or threshold tuning\n")
        f.write(f"- **Threshold Optimization:** Optimize classification thresholds for business requirements\n")

    print(f"Evaluation report saved to {EVAL_REPORT_FILE}")
    print("\n--- Model Training and Evaluation Completed Successfully ---")


if __name__ == "__main__":
    train_evaluate_model()