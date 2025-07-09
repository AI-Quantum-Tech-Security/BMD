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

warnings.filterwarnings('ignore')

# --- Global Configuration ---
DATA_FILE = 'files/synthetic_behavioral_data.csv'
MODEL_OUTPUT_FILE = 'files/model.pkl'
TARGET_COLUMN = 'risk_label'
EVAL_REPORT_FILE = 'files/risk_model_eval.md'
MODEL_FEATURES_FILE = 'files/model_features.json'

# --- Enhanced Feature Definitions ---
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


def add_noise_to_features(X, noise_level=0.01):
    """
    Dodaje szum do cech numerycznych aby uniknąć idealnej separacji
    """
    X_noisy = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in X_noisy.columns:
            noise = np.random.normal(0, noise_level * X_noisy[col].std(), size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise

    return X_noisy


def create_feature_interactions(X):
    """
    Tworzy nowe cechy jako interakcje między istniejącymi
    """
    X_enhanced = X.copy()

    # Przykładowe interakcje dla danych behawioralnych
    if 'tx_amount' in X.columns and 'avg_tx_amount' in X.columns:
        X_enhanced['tx_amount_deviation'] = abs(X['tx_amount'] - X['avg_tx_amount'])

    if 'session_duration' in X.columns and 'txs_last_24h' in X.columns:
        X_enhanced['tx_intensity'] = X['txs_last_24h'] / (X['session_duration'] + 1)

    if 'device_change_freq' in X.columns and 'location_change_freq' in X.columns:
        X_enhanced['behavior_volatility'] = X['device_change_freq'] + X['location_change_freq']

    return X_enhanced


def calculate_multiclass_roc_auc(y_true, y_pred_proba, classes):
    """
    Calculate ROC-AUC for multiclass classification using one-vs-rest approach
    """
    try:
        y_true_binarized = label_binarize(y_true, classes=classes)

        if len(classes) == 2:
            y_true_binarized = np.column_stack([1 - y_true_binarized, y_true_binarized])

        roc_auc_per_class = {}
        for i, class_name in enumerate(classes):
            if len(classes) == 2 and i == 0:
                continue
            try:
                auc_score = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
                roc_auc_per_class[class_name] = auc_score
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC for class {class_name}: {e}")
                roc_auc_per_class[class_name] = None

        valid_aucs = [auc for auc in roc_auc_per_class.values() if auc is not None]

        if valid_aucs:
            macro_auc = np.mean(valid_aucs)
            weighted_auc = macro_auc
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


def compare_models(X_train, X_test, y_train, y_test):
    """
    Porównuje różne modele ML
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=50,  # Zmniejszone żeby ograniczyć overfitting
            max_depth=10,  # Ograniczona głębokość
            min_samples_split=20,  # Większa minimalna liczba próbek
            min_samples_leaf=10,  # Większa minimalna liczba liści
            random_state=42,
            class_weight='balanced'
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        'SVM': SVC(
            random_state=42,
            class_weight='balanced',
            probability=True,
            kernel='rbf'
        )
    }

    results = {}

    # Normalizacja danych dla SVM i Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        print(f"\nTrening modelu: {name}")

        if name in ['LogisticRegression', 'SVM']:
            # Używaj znormalizowanych danych
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            # RandomForest nie potrzebuje normalizacji
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled if name in ['LogisticRegression', 'SVM'] else X_train,
                                    y_train, cv=5, scoring='f1_weighted')

        # Metryki
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC-AUC
        classes = model.classes_
        roc_auc_per_class, macro_auc, weighted_auc = calculate_multiclass_roc_auc(y_test, y_pred_proba, classes)

        results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': macro_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'scaler': scaler if name in ['LogisticRegression', 'SVM'] else None
        }

        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {macro_auc:.4f}" if macro_auc else "  ROC-AUC: Could not calculate")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return results


def train_evaluate_model():
    print(f"--- Starting Enhanced Model Training and Evaluation ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run data generation first.")
        return

    df = pd.read_csv(DATA_FILE)
    print("Data loaded for model training.")
    print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

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

    # Sprawdź czy dane są zbyt czytelne
    class_separation = df.groupby(TARGET_COLUMN).mean()
    print(f"\nMean values by class (sprawdzamy separację):")
    print(class_separation.head())

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

    print("\nStarting enhanced data preprocessing...")

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

    # Process categorical features z ograniczeniem kardynalności
    current_categorical_features_in_X = [col for col in CATEGORICAL_FEATURES_FOR_MODEL if col in X.columns]
    if current_categorical_features_in_X:
        print(f"Applying One-Hot Encoding to categorical features: {current_categorical_features_in_X}")

        # Ograniczamy kardynalność kategorycznych cech
        for col in current_categorical_features_in_X:
            value_counts = X[col].value_counts()
            if len(value_counts) > 10:  # Jeśli więcej niż 10 unikalnych wartości
                top_values = value_counts.head(9).index.tolist()
                X[col] = X[col].apply(lambda x: x if x in top_values else 'Other')
                print(f"Reduced cardinality for {col} to top 9 values + 'Other'")

        X = pd.get_dummies(X, columns=current_categorical_features_in_X, drop_first=True)
        print(f"Shape after One-Hot Encoding: {X.shape}")

    # Handle missing values
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64'] and X[col].isnull().any():
            mean_val = X[col].mean()
            X[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in numerical column '{col}' with mean: {mean_val:.4f}")

    # Feature engineering - dodanie interakcji
    print("Creating feature interactions...")
    X = create_feature_interactions(X)
    print(f"Shape after feature interactions: {X.shape}")

    # Dodanie szumu żeby uniknąć perfect separation
    print("Adding noise to prevent overfitting...")
    X = add_noise_to_features(X, noise_level=0.02)

    # Check for non-numeric columns
    non_numeric_cols_after_prep = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols_after_prep) > 0:
        print(f"Error: Non-numeric columns found after preprocessing: {non_numeric_cols_after_prep}")
        return

    print("Enhanced data preprocessing completed.")

    # Save the columns after preprocessing
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
        X, y, test_size=0.3, random_state=42, stratify=y  # Zwiększony test set
    )

    print(f"\nTraining set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")
    print(f"Target distribution in training set:")
    print(y_train.value_counts())
    print(f"Target distribution in test set:")
    print(y_test.value_counts())

    # Sprawdź czy klasy są zbalansowane - jeśli nie, użyj SMOTE
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    if imbalance_ratio > 2:
        print(f"\nDetected class imbalance (ratio: {imbalance_ratio:.2f}). Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Training set size: {X_train_balanced.shape[0]} rows")
        print(f"New class distribution:")
        print(pd.Series(y_train_balanced).value_counts())
        X_train, y_train = X_train_balanced, y_train_balanced

    # Compare different models
    print("\n=== COMPARING DIFFERENT MODELS ===")
    model_results = compare_models(X_train, X_test, y_train, y_test)

    # Select best model based on cross-validation F1 score
    best_model_name = max(model_results.keys(),
                          key=lambda x: model_results[x]['cv_mean'])
    best_model_info = model_results[best_model_name]
    best_model = best_model_info['model']

    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Cross-validation F1: {best_model_info['cv_mean']:.4f} (+/- {best_model_info['cv_std'] * 2:.4f})")

    # Detailed evaluation of best model
    y_pred = best_model_info['y_pred']
    y_pred_proba = best_model_info['y_pred_proba']
    classes = best_model.classes_

    print(f"Model classes: {classes}")

    # Calculate metrics
    roc_auc_per_class, macro_auc, weighted_auc = calculate_multiclass_roc_auc(y_test, y_pred_proba, classes)

    if roc_auc_per_class:
        print("\nROC-AUC per class (One-vs-Rest):")
        for class_name, auc_score in roc_auc_per_class.items():
            if auc_score is not None:
                print(f"  {class_name}: {auc_score:.4f}")
            else:
                print(f"  {class_name}: Could not calculate")

    if macro_auc is not None:
        print(f"Macro-average ROC-AUC: {macro_auc:.4f}")
    else:
        print("Could not calculate macro/weighted ROC-AUC")

    # Confusion Matrix
    print("\n=== CONFUSION MATRIX ===")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Confusion Matrix classes order: {classes}")

    # Per-class metrics
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

    # Feature Importance (tylko dla Random Forest)
    print("\n=== FEATURE IMPORTANCE ===")
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(best_model.feature_importances_,
                                        index=X.columns).sort_values(ascending=False)
        print("Top 15 most important features:")
        print(feature_importances.head(15))
    else:
        print("Feature importance not available for this model type")

    # Save best model
    print(f"\nSaving best model ({best_model_name}) to file: {MODEL_OUTPUT_FILE}")
    os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)
    try:
        model_to_save = {
            'model': best_model,
            'model_type': best_model_name,
            'scaler': best_model_info['scaler'],
            'features': X.columns.tolist()
        }
        joblib.dump(model_to_save, MODEL_OUTPUT_FILE)
        print(f"Model successfully saved as {MODEL_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

    # Generate enhanced evaluation report
    print(f"\nGenerating enhanced evaluation report: {EVAL_REPORT_FILE}")
    os.makedirs(os.path.dirname(EVAL_REPORT_FILE), exist_ok=True)

    with open(EVAL_REPORT_FILE, 'w') as f:
        f.write(f"# Enhanced Risk Model Evaluation Report\n\n")

        f.write(f"## Model Comparison Results\n")
        f.write(f"| Model | F1-Score | ROC-AUC | CV Mean | CV Std |\n")
        f.write(f"|-------|----------|---------|---------|--------|\n")
        for name, results in model_results.items():
            f.write(
                f"| {name} | {results['f1']:.4f} | {results['roc_auc']:.4f if results['roc_auc'] else 'N/A'} | {results['cv_mean']:.4f} | {results['cv_std']:.4f} |\n")
        f.write(f"\n**Best Model: {best_model_name}** ⭐\n\n")

        f.write(f"## Model Details\n")
        f.write(f"- **Best Model Type:** {best_model_name}\n")
        f.write(
            f"- **Cross-validation F1:** {best_model_info['cv_mean']:.4f} (+/- {best_model_info['cv_std'] * 2:.4f})\n")
        f.write(f"- **Model Classes:** {list(classes)}\n\n")

        f.write(f"## Data Overview\n")
        f.write(f"- **Total rows in dataset:** {df.shape[0]}\n")
        f.write(f"- **Training set rows:** {X_train.shape[0]}\n")
        f.write(f"- **Test set rows:** {X_test.shape[0]}\n")
        f.write(f"- **Features after preprocessing:** {X.shape[1]}\n")
        f.write(f"- **Target column:** `{TARGET_COLUMN}`\n")
        f.write(f"- **Class imbalance ratio:** {imbalance_ratio:.2f}\n\n")

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
            f.write(f"**Macro-average ROC-AUC:** {macro_auc:.4f}\n\n")

        # Per-class metrics
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
        if hasattr(best_model, 'feature_importances_'):
            f.write(f"### Feature Importance (Top 15)\n")
            f.write(f"```\n")
            f.write(f"{feature_importances.head(15).to_string()}\n")
            f.write(f"```\n\n")

        # Enhanced analysis
        f.write(f"## Enhanced Analysis Summary\n")

        # Performance assessment
        if macro_auc is not None:
            f.write(f"- **Overall ROC-AUC Performance:** {macro_auc:.4f}\n")
            if macro_auc >= 0.9:
                f.write(f"  - ⚠️ Bardzo wysoki AUC - sprawdź czy nie ma overfittingu\n")
            elif macro_auc >= 0.8:
                f.write(f"  - ✅ Dobra wydajność dyskryminacyjna\n")
            elif macro_auc >= 0.7:
                f.write(f"  - ⚠️ Umiarkowana wydajność dyskryminacyjna\n")
            else:
                f.write(f"  - ❌ Słaba wydajność - model wymaga poprawy\n")

        # Cross-validation stability
        cv_stability = best_model_info['cv_std'] / best_model_info['cv_mean']
        f.write(f"- **Model Stability (CV):** {cv_stability:.4f}\n")
        if cv_stability < 0.1:
            f.write(f"  - ✅ Stabilny model\n")
        elif cv_stability < 0.2:
            f.write(f"  - ⚠️ Umiarkowanie stabilny\n")
        else:
            f.write(f"  - ❌ Niestabilny model - wysoka wariancja\n")

        f.write(f"\n## Recommendations for Further Improvement\n")
        f.write(f"- **More Data:** Zbierz więcej różnorodnych danych treningowych\n")
        f.write(f"- **Feature Engineering:** Dodaj więcej cech behawioralnych\n")
        f.write(f"- **Ensemble Methods:** Rozważ XGBoost, LightGBM\n")
        f.write(f"- **Regularization:** Zwiększ regularyzację aby uniknąć overfittingu\n")
        f.write(f"- **Time-based Validation:** Użyj time-series split dla walidacji\n")
        f.write(f"- **Threshold Tuning:** Zoptymalizuj progi klasyfikacji\n")
        f.write(f"- **Monitoring:** Implementuj monitoring drift'u danych\n")

    print(f"Enhanced evaluation report saved to {EVAL_REPORT_FILE}")
    print("\n--- Enhanced Model Training and Evaluation Completed Successfully ---")


if __name__ == "__main__":
    train_evaluate_model()