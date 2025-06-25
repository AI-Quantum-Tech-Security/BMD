import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
import joblib
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
import warnings

warnings.filterwarnings('ignore')

# === Configuration ===
DATA_FILE = 'files/synthetic_behavioral_data.csv'
MODEL_OUTPUT_FILE = 'files/risk_model.pkl'
MODEL_FEATURES_FILE = 'files/model_features.json'
EVAL_REPORT_FILE = 'files/risk_model_eval.md'
TARGET_COLUMN = 'risk_label'

# Enhanced data generation parameters
NUM_NORMAL_USERS = 2000  # Increased for more diversity
NUM_ANOMALOUS_USERS = 400  # Increased for better balance
TRANSACTIONS_PER_NORMAL_USER_MEAN = 75  # More transactions per user
TRANSACTIONS_PER_ANOMALOUS_USER_MEAN = 45
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Enhanced feature sets
NUMERIC_FEATURES = [
    'avg_tx_amount', 'device_change_freq', 'tx_hour', 'location_change_freq',
    'transaction_count_24h', 'time_since_last_tx', 'tx_amount_to_balance_ratio',
    'ip_address_reputation', 'transaction_velocity_10min', 'tx_amount', 'account_balance',
    'session_duration', 'login_frequency_7d', 'failed_login_attempts'
]

BOOLEAN_FEATURES = [
    'is_new_device', 'is_weekend', 'country_change_flag', 'is_vpn_detected',
    'is_high_risk_merchant', 'is_unusual_hour'
]


def add_realistic_noise_and_complexity(data, risk_label):
    """
    Add more realistic variations and edge cases to make data less perfectly separable
    """
    # Add some legitimate transactions that look suspicious
    if risk_label == "legit" and random.random() < 0.15:  # 15% of legit transactions have suspicious patterns
        if random.random() < 0.5:  # Large amount transaction
            data[15] = data[15] * random.uniform(3, 6)  # tx_amount
        else:  # Unusual time
            data[2] = random.choice([1, 2, 3, 23])  # tx_hour

    # Add some fraud transactions that look normal
    if risk_label == "fraud" and random.random() < 0.25:  # 25% of fraud looks normal
        data[2] = random.randint(9, 17)  # Normal business hours
        data[8] = random.uniform(0.7, 0.95)  # Good IP reputation
        data[4] = 0  # Not a new device

    # Add some suspicious transactions that are actually legitimate
    if risk_label == "suspicious" and random.random() < 0.30:  # 30% are false positives
        data[8] = random.uniform(0.8, 1.0)  # Good IP reputation
        data[11] = 0  # No country change

    return data


def generate_enhanced_behavioral_data():
    """
    Generate more realistic and complex behavioral data with better separation challenges
    """
    print("--- Starting enhanced realistic synthetic data generation ---")
    data = []
    user_id_counter = 1
    tx_id_counter = 1

    # Generate merchant and location pools for more realistic categorical features
    merchants = [f'merchant_{i}' for i in range(1, 201)]  # 200 merchants
    locations = ['US', 'UK', 'DE', 'FR', 'IT', 'ES', 'CA', 'AU', 'JP', 'CN', 'IN', 'BR', 'MX']
    device_types = ['mobile', 'desktop', 'tablet', 'unknown']

    print(f"Generating data for {NUM_NORMAL_USERS} normal users...")
    for user_idx in range(NUM_NORMAL_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1

        # User profile - more realistic distributions
        user_avg_tx_amount = np.random.lognormal(mean=4, sigma=1)  # Log-normal distribution
        user_device_change_freq = np.random.beta(2, 98)  # Most users rarely change devices
        user_location_change_freq = np.random.beta(1, 199)  # Even rarer location changes
        user_account_balance = np.random.lognormal(mean=8, sigma=1)
        user_preferred_merchant = random.choice(merchants)
        user_primary_location = random.choice(locations)
        user_device_type = random.choice(device_types)

        # Behavioral patterns
        user_login_pattern = random.choice(['morning', 'afternoon', 'evening', 'mixed'])
        user_session_duration_mean = np.random.uniform(5, 60)  # minutes

        num_transactions = max(10, int(np.random.negative_binomial(TRANSACTIONS_PER_NORMAL_USER_MEAN, 0.3)))
        user_transactions = []

        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            risk_label = "legit"

            # Generate features based on user profile and some randomness
            if user_login_pattern == 'morning':
                tx_hour = max(6, min(12, int(np.random.normal(9, 2))))
            elif user_login_pattern == 'afternoon':
                tx_hour = max(12, min(18, int(np.random.normal(15, 2))))
            elif user_login_pattern == 'evening':
                tx_hour = max(18, min(23, int(np.random.normal(20, 2))))
            else:  # mixed
                tx_hour = random.randint(6, 23)

            is_weekend = 1 if timestamp.weekday() >= 5 else 0
            is_unusual_hour = 1 if tx_hour < 6 or tx_hour > 23 else 0

            # Transaction amount with user preference + some variance
            tx_amount = max(1, np.random.normal(user_avg_tx_amount, user_avg_tx_amount * 0.4))

            # Device and location patterns
            is_new_device = 1 if random.random() < user_device_change_freq else 0
            country_change_flag = 1 if random.random() < user_location_change_freq else 0

            # IP and security features
            ip_address_reputation = np.random.beta(8, 2)  # Most IPs are good
            is_vpn_detected = 1 if random.random() < 0.05 else 0

            # Session and timing features
            session_duration = max(1, np.random.normal(user_session_duration_mean, 15))
            time_since_last_tx = 0 if i == 0 else max(0.1, np.random.exponential(24))  # hours

            # Behavioral aggregates
            transaction_count_24h = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 86400])
            transaction_velocity_10min = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 600])
            login_frequency_7d = max(1, int(np.random.normal(15, 5)))
            failed_login_attempts = np.random.poisson(0.5)

            # Risk calculations
            tx_amount_to_balance_ratio = min(1.0, tx_amount / user_account_balance) if user_account_balance > 0 else 0
            is_high_risk_merchant = 1 if random.random() < 0.02 else 0

            # Store transaction for temporal features
            user_transactions.append({'timestamp': timestamp})

            # Create data row
            data_row = [
                user_avg_tx_amount, user_device_change_freq, tx_hour, user_location_change_freq,
                is_new_device, transaction_count_24h, time_since_last_tx, tx_amount_to_balance_ratio,
                ip_address_reputation, is_weekend, transaction_velocity_10min, country_change_flag,
                user_id, tx_id_counter, timestamp, tx_amount, user_account_balance, risk_label, 0,
                session_duration, login_frequency_7d, failed_login_attempts, is_vpn_detected,
                is_high_risk_merchant, is_unusual_hour, user_preferred_merchant, user_primary_location, user_device_type
            ]

            # Add realistic noise
            data_row = add_realistic_noise_and_complexity(data_row, risk_label)
            data.append(data_row)
            tx_id_counter += 1

    print(f"Generating data for {NUM_ANOMALOUS_USERS} anomalous users...")
    for user_idx in range(NUM_ANOMALOUS_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1

        # Anomalous user profiles - different patterns
        user_avg_tx_amount = np.random.lognormal(mean=5.5, sigma=1.2)  # Higher amounts
        user_device_change_freq = np.random.beta(15, 85)  # More device changes
        user_location_change_freq = np.random.beta(10, 90)  # More location changes
        user_account_balance = np.random.lognormal(mean=7, sigma=1.5)
        user_preferred_merchant = random.choice(merchants)
        user_primary_location = random.choice(locations)
        user_device_type = random.choice(device_types)

        user_session_duration_mean = np.random.uniform(1, 10)  # Shorter sessions

        num_transactions = max(5, int(np.random.negative_binomial(TRANSACTIONS_PER_ANOMALOUS_USER_MEAN, 0.4)))
        user_transactions = []

        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            is_weekend = 1 if timestamp.weekday() >= 5 else 0

            # Determine risk level with more nuanced distribution
            rand_val = random.random()
            if rand_val < 0.30:  # 30% fraud
                risk_label = "fraud"
            elif rand_val < 0.65:  # 35% suspicious
                risk_label = "suspicious"
            else:  # 35% legit (false positives)
                risk_label = "legit"

            risk_flag_manual = 0 if risk_label == "legit" else 1

            # Generate features based on risk level but with more overlap
            if risk_label == "fraud":
                tx_hour = np.random.choice([0, 1, 2, 3, 4, 5, 23], p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1])
                tx_amount = max(100, user_account_balance * np.random.uniform(0.3, 1.5))
                is_new_device = 1 if random.random() < 0.70 else 0  # Reduced from 0.9
                country_change_flag = 1 if random.random() < 0.60 else 0  # Reduced from 0.85
                ip_address_reputation = np.random.beta(1, 4)  # Low reputation
                is_vpn_detected = 1 if random.random() < 0.40 else 0
                session_duration = max(0.5, np.random.exponential(3))
                failed_login_attempts = np.random.poisson(3)
                is_high_risk_merchant = 1 if random.random() < 0.25 else 0

            elif risk_label == "suspicious":
                tx_hour = np.random.choice([6, 7, 22, 23, 0, 1], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
                tx_amount = max(1, user_avg_tx_amount * np.random.uniform(1.5, 3.0))  # Reduced multiplier
                is_new_device = 1 if random.random() < 0.25 else 0  # Reduced from 0.4
                country_change_flag = 1 if random.random() < 0.15 else 0  # Reduced from 0.2
                ip_address_reputation = np.random.beta(2, 3)  # Medium reputation
                is_vpn_detected = 1 if random.random() < 0.15 else 0
                session_duration = max(1, np.random.normal(8, 5))
                failed_login_attempts = np.random.poisson(1)
                is_high_risk_merchant = 1 if random.random() < 0.10 else 0

            else:  # legit but flagged user
                tx_hour = random.randint(8, 20)
                tx_amount = max(1, np.random.normal(user_avg_tx_amount, user_avg_tx_amount * 0.6))
                is_new_device = 1 if random.random() < 0.10 else 0
                country_change_flag = 1 if random.random() < 0.05 else 0
                ip_address_reputation = np.random.beta(6, 2)  # Good reputation
                is_vpn_detected = 1 if random.random() < 0.08 else 0
                session_duration = max(1, np.random.normal(20, 10))
                failed_login_attempts = np.random.poisson(0.2)
                is_high_risk_merchant = 1 if random.random() < 0.03 else 0

            is_unusual_hour = 1 if tx_hour < 6 or tx_hour > 23 else 0

            # Temporal features
            time_since_last_tx = 0 if i == 0 else max(0.01, np.random.exponential(2))
            transaction_count_24h = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 86400])
            transaction_velocity_10min = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 600])
            login_frequency_7d = max(1, int(np.random.normal(8, 4)))

            tx_amount_to_balance_ratio = min(1.0, tx_amount / user_account_balance) if user_account_balance > 0 else 0

            user_transactions.append({'timestamp': timestamp})

            # Create data row
            data_row = [
                user_avg_tx_amount, user_device_change_freq, tx_hour, user_location_change_freq,
                is_new_device, transaction_count_24h, time_since_last_tx, tx_amount_to_balance_ratio,
                ip_address_reputation, is_weekend, transaction_velocity_10min, country_change_flag,
                user_id, tx_id_counter, timestamp, tx_amount, user_account_balance, risk_label, risk_flag_manual,
                session_duration, login_frequency_7d, failed_login_attempts, is_vpn_detected,
                is_high_risk_merchant, is_unusual_hour, user_preferred_merchant, user_primary_location, user_device_type
            ]

            # Add realistic noise
            data_row = add_realistic_noise_and_complexity(data_row, risk_label)
            data.append(data_row)
            tx_id_counter += 1

    # Enhanced columns
    columns = [
        'avg_tx_amount', 'device_change_freq', 'tx_hour', 'location_change_freq', 'is_new_device',
        'transaction_count_24h', 'time_since_last_tx', 'tx_amount_to_balance_ratio', 'ip_address_reputation',
        'is_weekend', 'transaction_velocity_10min', 'country_change_flag', 'user_id', 'tx_id', 'timestamp',
        'tx_amount', 'account_balance', 'risk_label', 'risk_flag_manual', 'session_duration',
        'login_frequency_7d', 'failed_login_attempts', 'is_vpn_detected', 'is_high_risk_merchant',
        'is_unusual_hour', 'merchant_id', 'location', 'device_type'
    ]

    df = pd.DataFrame(data, columns=columns)

    print(f"\nSaving enhanced data to file: {DATA_FILE}")
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

    print(f"File {DATA_FILE} successfully generated.")
    print(f"Generated {df.shape[0]} rows of data with {df.shape[1]} features.")
    print(f"'{TARGET_COLUMN}' distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"'{TARGET_COLUMN}' distribution (proportions):")
    print(df[TARGET_COLUMN].value_counts(normalize=True))

    # Data quality checks
    print(f"\nData Quality Checks:")
    print(f"Missing values per column:")
    print(df.isnull().sum().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Feature correlation with target (potential data leakage check)
    print(f"\nChecking for potential data leakage...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corrwith(df['risk_flag_manual']).abs().sort_values(ascending=False)
    high_corr = correlations[correlations > 0.8]
    if len(high_corr) > 1:  # Exclude perfect correlation with itself
        print(f"⚠️ Warning: High correlations detected (>0.8): {high_corr.head()}")
    else:
        print(f"✅ No suspicious high correlations detected")

    print("\n--- Enhanced data generation completed ---\n")
    return df


def create_feature_interactions(X):
    """
    Create meaningful feature interactions for behavioral data
    """
    X_enhanced = X.copy()

    # Risk interaction features
    if 'tx_amount' in X.columns and 'account_balance' in X.columns:
        X_enhanced['amount_balance_ratio'] = X['tx_amount'] / (X['account_balance'] + 1)

    if 'device_change_freq' in X.columns and 'location_change_freq' in X.columns:
        X_enhanced['mobility_score'] = X['device_change_freq'] + X['location_change_freq']

    if 'transaction_velocity_10min' in X.columns and 'session_duration' in X.columns:
        X_enhanced['tx_intensity'] = X['transaction_velocity_10min'] / (X['session_duration'] + 1)

    if 'failed_login_attempts' in X.columns and 'is_new_device' in X.columns:
        X_enhanced['auth_risk_score'] = X['failed_login_attempts'] * (X['is_new_device'] + 1)

    if 'tx_hour' in X.columns:
        # Time-based features
        X_enhanced['is_business_hours'] = ((X['tx_hour'] >= 9) & (X['tx_hour'] <= 17)).astype(int)
        X_enhanced['is_late_night'] = ((X['tx_hour'] >= 23) | (X['tx_hour'] <= 5)).astype(int)

    return X_enhanced


def train_evaluate_enhanced_model():
    """
    Enhanced model training with multiple algorithms and better evaluation
    """
    print(f"--- Starting enhanced model training and evaluation ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run the generator first.")
        return

    df = pd.read_csv(DATA_FILE)
    print("Data loaded for model training.")
    print(f"Dataset shape: {df.shape}")

    # Enhanced feature selection
    model_features_list = NUMERIC_FEATURES + BOOLEAN_FEATURES

    # Handle categorical features
    categorical_features = ['merchant_id', 'location', 'device_type']
    for cat_col in categorical_features:
        if cat_col in df.columns:
            # Limit cardinality and create dummy variables
            value_counts = df[cat_col].value_counts()
            if len(value_counts) > 15:
                top_values = value_counts.head(14).index.tolist()
                df[cat_col] = df[cat_col].apply(lambda x: x if x in top_values else 'Other')

            # Create dummy variables
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            model_features_list.extend(dummies.columns.tolist())

    # Filter existing features
    existing_features = [col for col in model_features_list if col in df.columns]
    X = df[existing_features].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\nFeatures used for training ({len(existing_features)}): {existing_features[:10]}...")

    # Create feature interactions
    print("Creating feature interactions...")
    X = create_feature_interactions(X)
    print(f"Shape after feature engineering: {X.shape}")

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['int64', 'float64']:
                fill_value = X[col].median()
                X[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with median: {fill_value:.4f}")
            else:
                fill_value = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
                X[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with mode: {fill_value}")

    print("Data preprocessing completed.")

    # Save feature names
    print(f"Saving feature names to file: {MODEL_FEATURES_FILE}")
    os.makedirs(os.path.dirname(MODEL_FEATURES_FILE), exist_ok=True)
    with open(MODEL_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(X.columns.tolist(), f, indent=2)

    # Enhanced train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")
    print(f"Class distribution in training set:")
    print(y_train.value_counts(normalize=True))

    # Check for class imbalance and apply SMOTE if needed
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    if imbalance_ratio > 1.5:
        print(f"\nApplying SMOTE for class imbalance (ratio: {imbalance_ratio:.2f})...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape[0]} rows")
        print(f"New class distribution:")
        print(pd.Series(y_train_balanced).value_counts(normalize=True))
        X_train, y_train = X_train_balanced, y_train_balanced

    # Define models with regularization to prevent overfitting
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Limited depth
            min_samples_split=10,  # Higher minimum
            min_samples_leaf=5,  # Higher minimum
            max_features='sqrt',  # Feature subsampling
            random_state=42,
            class_weight='balanced',
            oob_score=True
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
            C=0.1,  # Stronger regularization
            solver='liblinear'
        ),
        'SVM': SVC(
            random_state=42,
            class_weight='balanced',
            probability=True,
            kernel='rbf',
            C=0.1,  # Stronger regularization
            gamma='scale'
        )
    }

    # Train and evaluate models
    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        print(f"\n=== Training {name} ===")

        # Use scaled data for LogReg and SVM
        if name in ['LogisticRegression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                        scoring='f1_weighted')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                        scoring='f1_weighted')

        # Calculate metrics
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # ROC-AUC
        classes = model.classes_
        roc_auc_per_class, macro_auc, _ = calculate_multiclass_roc_auc(y_test, y_pred_proba, classes)

        results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': macro_auc,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {macro_auc:.4f}" if macro_auc else "ROC-AUC: Could not calculate")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        if hasattr(model, 'oob_score_'):
            print(f"OOB Score: {model.oob_score_:.4f}")

    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']
    best_results = results[best_model_name]

    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Cross-validation F1: {best_results['cv_mean']:.4f} (+/- {best_results['cv_std'] * 2:.4f})")

    # Detailed evaluation
    y_pred = best_results['y_pred']
    y_pred_proba = best_results['y_pred_proba']
    classes = best_model.classes_

    print("\n=== Confusion Matrix ===")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    print(pd.DataFrame(conf_matrix, index=classes, columns=[f'Predicted {c}' for c in classes]))

    print("\n=== Classification Report ===")
    class_report = classification_report(y_test, y_pred, labels=classes)
    print(class_report)

    roc_auc_per_class, macro_auc, _ = calculate_multiclass_roc_auc(y_test, y_pred_proba, classes)
    print("\n=== ROC-AUC Results ===")
    if roc_auc_per_class:
        for class_name, auc_score in roc_auc_per_class.items():
            if auc_score is not None:
                print(f"  ROC-AUC for class '{class_name}': {auc_score:.4f}")

    # Save best model
    print(f"\nSaving best model ({best_model_name}) to file: {MODEL_OUTPUT_FILE}")
    model_package = {
        'model': best_model,
        'model_type': best_model_name,
        'scaler': scaler if best_model_name in ['LogisticRegression', 'SVM'] else None,
        'feature_names': X.columns.tolist(),
        'classes': classes.tolist()
    }
    joblib.dump(model_package, MODEL_OUTPUT_FILE)
    print("Model successfully saved.")

    # Generate enhanced evaluation report
    print(f"\nGenerating enhanced evaluation report: {EVAL_REPORT_FILE}")
    try:
        with open(EVAL_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write("# Enhanced Behavioral Risk Model Evaluation Report\n\n")

            # Model comparison table
            f.write("## Model Comparison Results\n\n")
            f.write("| Model | F1-Score | ROC-AUC | CV Mean | CV Std | Status |\n")
            f.write("|-------|----------|---------|---------|--------|---------|\n")
            for name, res in results.items():
                status = "✅ Best" if name == best_model_name else ""
                f.write(
                    f"| {name} | {res['f1']:.4f} | {res['roc_auc']:.4f if res['roc_auc'] else 'N/A'} | {res['cv_mean']:.4f} | {res['cv_std']:.4f} | {status} |\n")

            f.write(f"\n## Best Model: {best_model_name}\n\n")
            f.write(
                f"- **Cross-validation F1:** {best_results['cv_mean']:.4f} (+/- {best_results['cv_std'] * 2:.4f})\n")
            f.write(f"- **Test F1-Score:** {best_results['f1']:.4f}\n")
            f.write(f"- **Test ROC-AUC:** {best_results['roc_auc']:.4f}\n")

            if hasattr(best_model, 'oob_score_'):
                f.write(f"- **Out-of-Bag Score:** {best_model.oob_score_:.4f}\n")

            f.write(f"\n### Confusion Matrix\n")
            f.write("```\n")
            f.write(str(pd.DataFrame(conf_matrix, index=classes, columns=[f'Predicted {c}' for c in classes])))
            f.write("\n```\n\n")

            f.write("### Classification Report\n")
            f.write("```\n")
            f.write(class_report)
            f.write("\n```\n\n")

            # Feature importance for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                f.write("### Feature Importance (Top 20)\n")
                feature_importances = pd.Series(best_model.feature_importances_,
                                                index=X.columns).sort_values(ascending=False)
                f.write("```\n")
                f.write(feature_importances.head(20).to_string())
                f.write("\n```\n\n")

            # Model analysis
            f.write("## Model Analysis\n\n")

            stability = best_results['cv_std'] / best_results['cv_mean']
            f.write(f"- **Model Stability:** {stability:.4f}")
            if stability < 0.1:
                f.write(" ✅ Very stable\n")
            elif stability < 0.2:
                f.write(" ⚠️ Moderately stable\n")
            else:
                f.write(" ❌ Unstable - high variance\n")

            if best_results['roc_auc'] and best_results['roc_auc'] > 0.95:
                f.write("- **Overfitting Check:** ⚠️ Very high AUC - monitor for overfitting\n")
            elif best_results['roc_auc'] and best_results['roc_auc'] > 0.8:
                f.write("- **Performance:** ✅ Good discriminative performance\n")
            else:
                f.write("- **Performance:** ⚠️ Moderate performance - consider improvements\n")

            f.write(f"\n## Recommendations\n")
            f.write(f"- **Data Collection:** Collect more diverse real-world examples\n")
            f.write(f"- **Feature Engineering:** Add more temporal and behavioral features\n")
            f.write(f"- **Model Monitoring:** Implement drift detection for production\n")
            f.write(f"- **Threshold Tuning:** Optimize decision thresholds for business metrics\n")
            f.write(f"- **Ensemble Methods:** Consider stacking multiple models\n")

        print("Enhanced evaluation report successfully saved.")
    except Exception as e:
        print(f"An error occurred while generating the report: {e}")

    print("\n--- Enhanced training and evaluation completed successfully ---")
    return results


def calculate_multiclass_roc_auc(y_true, y_pred_proba, classes):
    """Calculate ROC-AUC for multiclass with better error handling"""
    try:
        y_true_binarized = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            y_true_binarized = np.column_stack([1 - y_true_binarized, y_true_binarized])

        roc_auc_per_class = {}
        for i, class_name in enumerate(classes):
            try:
                auc_score = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
                roc_auc_per_class[class_name] = auc_score
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC for class {class_name}: {e}")
                roc_auc_per_class[class_name] = None

        valid_aucs = [auc for auc in roc_auc_per_class.values() if auc is not None]
        macro_auc = np.mean(valid_aucs) if valid_aucs else None
        return roc_auc_per_class, macro_auc, macro_auc
    except Exception as e:
        print(f"Error in ROC-AUC calculation: {e}")
        return {}, None, None


if __name__ == "__main__":
    print("=== Enhanced Behavioral Risk Model Pipeline ===\n")

    # Step 1: Generate enhanced data
    print("Step 1: Generating enhanced synthetic data...")
    df = generate_enhanced_behavioral_data()

    # Step 2: Train and evaluate models
    print("Step 2: Training and evaluating models...")
    model_results = train_evaluate_enhanced_model()

    print("\n=== Pipeline completed successfully! ===")
    print(f"📊 Check the results in: {EVAL_REPORT_FILE}")
    print(f"🤖 Model saved to: {MODEL_OUTPUT_FILE}")
    print(f"📋 Features saved to: {MODEL_FEATURES_FILE}")

    # Summary statistics
    if model_results:
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['cv_mean'])
        best_score = model_results[best_model]['cv_mean']
        print(f"🏆 Best model: {best_model} (CV F1: {best_score:.4f})")

        if best_score > 0.95:
            print("⚠️  Warning: Very high performance - check for data leakage or overfitting")
        elif best_score > 0.8:
            print("✅ Good model performance achieved")
        else:
            print("📈 Model performance is moderate - consider improvements")