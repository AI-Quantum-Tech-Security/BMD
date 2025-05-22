import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import joblib
import os
import json # Import json module

#---Global Configuration---
DATA_FILE = 'synthetic_behavioral_data.csv' # Using the harder data
MODEL_OUTPUT_FILE = 'risk_model.pkl'
TARGET_COLUMN = 'risk_flag_manual'
EVAL_REPORT_FILE = 'risk_model_eval.md'
MODEL_FEATURES_FILE = 'model_features.json' # New file to store feature names

#---Feature Definitions for Model Preprocessing---
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
        return
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

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

    current_categorical_features_in_X = [col for col in CATEGORICAL_FEATURES_FOR_MODEL if col in X.columns]
    if current_categorical_features_in_X:
        print(f"Applying One-Hot Encoding to categorical features: {current_categorical_features_in_X}")
        X = pd.get_dummies(X, columns=current_categorical_features_in_X, drop_first=True)
        print(f"Shape after One-Hot Encoding: {X.shape}")

    for col in X.columns:
        if X[col].dtype in ['int64', 'float64'] and X[col].isnull().any():
            mean_val = X[col].mean()
            X[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in numerical column '{col}'")

    if X.isnull().values.any():
        print("Warning: Missing values still exist in the preprocessed data.")

    non_numeric_cols_after_prep = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols_after_prep) > 0:
        print(f"Error: Non-numeric columns found after preprocessing: {non_numeric_cols_after_prep}")
        return
    print("Data preprocessing completed.")

    # --- NEW: Save the columns after preprocessing ---
    print(f"Saving preprocessed feature column names to {MODEL_FEATURES_FILE}")
    try:
        with open(MODEL_FEATURES_FILE, 'w') as f:
            json.dump(X.columns.tolist(), f)
        print(f"Feature column names saved to {MODEL_FEATURES_FILE}")
    except Exception as e:
        print(f"Error occurred while saving feature column names: {e}")
    # --- END NEW ---

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nTraining set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")

    print("\nStarting Random Forest model training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training completed.")

    print("\nEvaluating model on the test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score on test set: {auc_score:.4f}")
    except ValueError as e:
        print(f"Could not calculate ROC-AUC. Error: {e}")
        auc_score = 'N/A' # Set to N/A if calculation fails

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Precision Score on test set: {precision:.4f}")
    print(f"Recall Score on test set: {recall:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(class_report)

    print(f"\nSaving trained model to file: {MODEL_OUTPUT_FILE}")
    try:
        joblib.dump(model, MODEL_OUTPUT_FILE)
        print(f"Model successfully saved as {MODEL_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

    print(f"\nGenerating evaluation report: {EVAL_REPORT_FILE}")
    with open(EVAL_REPORT_FILE, 'w') as f:
        f.write(f"# Risk Model Evaluation Report\n\n")
        f.write(f"## Model Details\n")
        f.write(f"-**Model Type:** RandomForestClassifier\n")
        f.write(f"-**Number of estimators:** {model.n_estimators}\n")
        f.write(f"-**Random state:** {model.random_state}\n\n")
        f.write(f"## Data Overview\n")
        f.write(f"-**Total rows in dataset:** {df.shape[0]}\n")
        f.write(f"-**Training set rows:** {X_train.shape[0]}\n")
        f.write(f"-**Test set rows:** {X_test.shape[0]}\n")
        f.write(f"-**Target column:** `{TARGET_COLUMN}`\n")
        f.write(f"-**Class distribution in test set:**\n")
        for class_val, proportion in y_test.value_counts(normalize=True).items():
            f.write(f"  -**Class {class_val}:** {proportion:.2%}\n")
        f.write(f"\n")
        f.write(f"## Evaluation Metrics on Test Set\n")
        f.write(f"-**ROC-AUC Score:** {auc_score:.4f}\n")
        f.write(f"-**Precision Score:** {precision:.4f}\n")
        f.write(f"-**Recall Score:** {recall:.4f}\n\n")
        f.write(f"### Confusion Matrix\n")
        f.write(f"```\n")
        f.write(f"{conf_matrix}\n")
        f.write(f"```\n\n")
        f.write(f"### Classification Report\n")
        f.write(f"```\n")
        f.write(f"{class_report}\n")
        f.write(f"```\n\n")
        f.write(f"## Feature Importance\n")
        try:
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            f.write(f"The top 10 most important features are:\n")
            f.write(f"```\n")
            f.write(f"{feature_importances.head(10).to_string()}\n")
            f.write(f"```\n\n")
        except Exception as e:
            f.write(f"Could not retrieve feature importances: {e}\n\n")

        f.write(f"## Next Steps & Considerations\n")
        f.write(f"-**Advanced Preprocessing:** Further explore preprocessing techniques.\n")
        f.write(f"-**Feature Engineering:** Create new features.\n")
        f.write(f"-**Class Imbalance:** Address class imbalance in the dataset.\n")
        f.write(f"-**Model Hyperparameter Tuning:** Optimize model parameters.\n")
        f.write(f"-**Cross-validation:** Implement robust model evaluation.\n")
        f.write(f"-**Alternative Models:** Experiment with other models.\n")
        f.write(f"-**Deployment:** Prepare the model for API deployment.\n")

    print(f"Evaluation report saved to {EVAL_REPORT_FILE}")
    print("\n--- Script finished execution ---")

if __name__ == "__main__":
    train_evaluate_model()