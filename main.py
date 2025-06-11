import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
import joblib
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

DATA_FILE = 'files/synthetic_behavioral_data.csv'
MODEL_OUTPUT_FILE = 'files/risk_model.pkl'
MODEL_FEATURES_FILE = 'files/model_features.json'
EVAL_REPORT_FILE = 'files/risk_model_eval.md'
TARGET_COLUMN = 'risk_label'

NUM_NORMAL_USERS = 1000
NUM_ANOMALOUS_USERS = 150
TRANSACTIONS_PER_NORMAL_USER_MEAN = 50
TRANSACTIONS_PER_ANOMALOUS_USER_MEAN = 30
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)

NUMERIC_FEATURES = [
    'avg_tx_amount', 'device_change_freq', 'tx_hour', 'location_change_freq',
    'transaction_count_24h', 'time_since_last_tx', 'tx_amount_to_balance_ratio',
    'ip_address_reputation', 'transaction_velocity_10min', 'tx_amount', 'account_balance'
]
BOOLEAN_FEATURES = ['is_new_device', 'is_weekend', 'country_change_flag']


def generate_behavioral_data():

    print("--- Starting realistic synthetic data generation ---")
    data = []
    user_id_counter = 1
    tx_id_counter = 1

    print(f"Generating data for {NUM_NORMAL_USERS} normal users...")
    for _ in range(NUM_NORMAL_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1
        user_avg_tx_amount = np.random.uniform(50, 800)
        user_device_change_freq = np.random.uniform(0, 0.01)
        user_location_change_freq = np.random.uniform(0, 0.02)
        user_account_balance = np.random.uniform(2000, 50000)
        num_transactions = max(5, int(np.random.normal(TRANSACTIONS_PER_NORMAL_USER_MEAN, 10)))
        user_transactions = []

        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            risk_label = "legit"
            risk_flag_manual = 0


            tx_hour = random.randint(8, 22)
            is_weekend = 1 if timestamp.weekday() >= 5 else 0
            tx_amount = max(1, np.random.normal(user_avg_tx_amount, user_avg_tx_amount * 0.2))

            is_new_device = 1 if random.random() < 0.005 else 0
            country_change_flag = 1 if random.random() < 0.001 else 0
            ip_address_reputation = np.random.uniform(0.85, 1.0)
            time_since_last_tx = 0 if i == 0 else max(0.1, abs((timestamp - user_transactions[-1][
                'timestamp']).total_seconds() / 3600))
            transaction_count_24h = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 86400])
            transaction_velocity_10min = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 600])
            tx_amount_to_balance_ratio = min(1.0, tx_amount / user_account_balance) if user_account_balance > 0 else 0

            user_transactions.append({'timestamp': timestamp})
            data.append([
                user_avg_tx_amount, user_device_change_freq, tx_hour, user_location_change_freq,
                is_new_device, transaction_count_24h, time_since_last_tx, tx_amount_to_balance_ratio,
                ip_address_reputation, is_weekend, transaction_velocity_10min, country_change_flag,
                user_id, tx_id_counter, timestamp, tx_amount, user_account_balance, risk_label, risk_flag_manual
            ])
            tx_id_counter += 1

    print(f"Generating data for {NUM_ANOMALOUS_USERS} anomalous users...")
    for _ in range(NUM_ANOMALOUS_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1
        user_avg_tx_amount = np.random.uniform(200, 2000)
        user_device_change_freq = np.random.uniform(0.1, 0.5)
        user_location_change_freq = np.random.uniform(0.1, 0.6)
        user_account_balance = np.random.uniform(100, 10000)
        num_transactions = max(5, int(np.random.normal(TRANSACTIONS_PER_ANOMALOUS_USER_MEAN, 8)))
        user_transactions = []

        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            is_weekend = 1 if timestamp.weekday() >= 5 else 0

            rand_val = random.random()
            if rand_val < 0.25:
                risk_label = "fraud"
            elif rand_val < 0.60:
                risk_label = "suspicious"
            else:
                risk_label = "legit"

            risk_flag_manual = 0 if risk_label == "legit" else 1

            if risk_label == "fraud":
                tx_hour = random.choice([0, 1, 2, 3, 4, 5, 23])
                tx_amount = max(1000, user_account_balance * np.random.uniform(0.7, 1.2))
                is_new_device = 1 if random.random() < 0.9 else 0
                country_change_flag = 1 if random.random() < 0.85 else 0
                ip_address_reputation = np.random.uniform(0.0, 0.4)
                time_since_last_tx = np.random.uniform(0.01, 0.05)
                transaction_velocity_10min = random.randint(3, 8)

            elif risk_label == "suspicious":
                tx_hour = random.choice([6, 7, 22, 23])
                tx_amount = max(1, user_avg_tx_amount * np.random.uniform(2.0, 4.5))
                is_new_device = 1 if random.random() < 0.4 else 0
                country_change_flag = 1 if random.random() < 0.2 else 0
                ip_address_reputation = np.random.uniform(0.2, 0.7)
                time_since_last_tx = np.random.uniform(0.05, 0.6)
                transaction_velocity_10min = random.randint(2, 4)

            else:  # risk_label == "legit"
                tx_hour = random.randint(9, 20)
                tx_amount = max(1, np.random.normal(user_avg_tx_amount, user_avg_tx_amount * 0.3))
                is_new_device = 1 if random.random() < 0.05 else 0
                country_change_flag = 0
                ip_address_reputation = np.random.uniform(0.6, 1.0)
                time_since_last_tx = 0 if i == 0 else max(0.5, abs((timestamp - user_transactions[-1][
                    'timestamp']).total_seconds() / 3600))
                transaction_velocity_10min = len(
                    [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 600])

            transaction_count_24h = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 86400])
            tx_amount_to_balance_ratio = min(1.0, tx_amount / user_account_balance) if user_account_balance > 0 else 0

            user_transactions.append({'timestamp': timestamp})
            data.append([
                user_avg_tx_amount, user_device_change_freq, tx_hour, user_location_change_freq,
                is_new_device, transaction_count_24h, time_since_last_tx, tx_amount_to_balance_ratio,
                ip_address_reputation, is_weekend, transaction_velocity_10min, country_change_flag,
                user_id, tx_id_counter, timestamp, tx_amount, user_account_balance, risk_label, risk_flag_manual
            ])
            tx_id_counter += 1

    columns = [
        'avg_tx_amount', 'device_change_freq', 'tx_hour', 'location_change_freq', 'is_new_device',
        'transaction_count_24h', 'time_since_last_tx', 'tx_amount_to_balance_ratio', 'ip_address_reputation',
        'is_weekend', 'transaction_velocity_10min', 'country_change_flag', 'user_id', 'tx_id', 'timestamp',
        'tx_amount', 'account_balance', 'risk_label', 'risk_flag_manual'
    ]
    df = pd.DataFrame(data, columns=columns)

    print(f"\nSaving data to file: {DATA_FILE}")
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"File {DATA_FILE} successfully generated.")
    print(f"Generated {df.shape[0]} rows of data.")
    print(f"'{TARGET_COLUMN}' distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")
    print("\n--- Data generation completed ---\n")


def train_evaluate_model():

    print(f"--- Starting model training and evaluation ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please run the generator first.")
        return

    df = pd.read_csv(DATA_FILE)
    print("Data loaded for model training.")

    model_features_list = NUMERIC_FEATURES + BOOLEAN_FEATURES
    X = df[model_features_list].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\nFeatures used for training ({len(model_features_list)}): {model_features_list}")

    for col in X.columns:
        if X[col].isnull().any():
            mean_val = X[col].mean()
            X[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in column '{col}' with mean: {mean_val:.4f}")

    print("\nData preprocessing completed.")

    print(f"Saving feature names to file: {MODEL_FEATURES_FILE}")
    os.makedirs(os.path.dirname(MODEL_FEATURES_FILE), exist_ok=True)
    with open(MODEL_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(X.columns.tolist(), f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

    print("\nStarting RandomForestClassifier model training...")
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=20,
        min_samples_leaf=5,
        oob_score=True
    )
    model.fit(X_train, y_train)
    print(f"Model training completed. OOB Score: {model.oob_score_:.4f}")

    print("\nEvaluating model on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    classes = model.classes_

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
    if macro_auc:
        print(f"  Macro-average ROC-AUC: {macro_auc:.4f}")

    print(f"\nSaving trained model to file: {MODEL_OUTPUT_FILE}")
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print("Model successfully saved.")

    print(f"\nGenerating evaluation report: {EVAL_REPORT_FILE}")
    try:
        with open(EVAL_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write("# Behavioral Risk Model Evaluation Report (Realistic Version)\n\n")
            f.write("## Metrics Summary\n\n")
            f.write(f"- **Macro-average ROC-AUC:** {macro_auc:.4f}\n")
            f.write(f"- **Accuracy:** {model.score(X_test, y_test):.4f}\n")
            f.write(f"- **Out-of-Bag (OOB) Score:** {model.oob_score_:.4f}\n\n")
            f.write("### Confusion Matrix\n")
            f.write("```\n")
            f.write(str(pd.DataFrame(conf_matrix, index=classes, columns=[f'Predicted {c}' for c in classes])))
            f.write("\n```\n\n")
            f.write("### Classification Report\n")
            f.write("```\n")
            f.write(class_report)
            f.write("\n```\n\n")
            f.write("### Feature Importance (Top 15)\n")
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            f.write("```\n")
            f.write(feature_importances.head(15).to_string())
            f.write("\n```\n")
        print("Evaluation report successfully saved.")
    except Exception as e:
        print(f"An error occurred while generating the report: {e}")

    print("\n--- Training and evaluation completed successfully ---")


def calculate_multiclass_roc_auc(y_true, y_pred_proba, classes):
    y_true_binarized = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_binarized = np.column_stack([1 - y_true_binarized, y_true_binarized])

    roc_auc_per_class = {}
    for i, class_name in enumerate(classes):
        try:
            auc_score = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc_per_class[class_name] = auc_score
        except ValueError:
            roc_auc_per_class[class_name] = None

    valid_aucs = [auc for auc in roc_auc_per_class.values() if auc is not None]
    macro_auc = np.mean(valid_aucs) if valid_aucs else None
    return roc_auc_per_class, macro_auc, macro_auc


if __name__ == "__main__":
    generate_behavioral_data()
    train_evaluate_model()

    print("\n=== Whole process completed successfully! ===")
    print(f"Check the results in the report file: {EVAL_REPORT_FILE}")


    # more information data enginnering combine in one
    # data should fit more model
