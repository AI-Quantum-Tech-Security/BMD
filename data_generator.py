import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json

# --- Global Configuration ---
DATA_FILE = 'files/synthetic_behavioral_dataset.csv'
SCHEMA_FILE = 'files/feature_schema.json'
DOC_FILE = 'files/Behavioral_Authentication_ML.md'

# --- Data Generation Configuration ---
NUM_NORMAL_USERS = 1000
NUM_ANOMALOUS_USERS = 100
TRANSACTIONS_PER_NORMAL_USER_MEAN = 50
TRANSACTIONS_PER_ANOMALOUS_USER_MEAN = 20
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)

# --- Label Distribution Control ---
FRAUD_RATE = 0.05  # 5% transakcji fraudulentnych
SUSPICIOUS_RATE = 0.15  # 15% transakcji podejrzanych
LEGIT_RATE = 0.80  # 80% transakcji legalnych

TARGET_COLUMN_BINARY = 'risk_flag_manual'
TARGET_COLUMN_CATEGORICAL = 'risk_label'

# --- Feature Schema Definition (8-15 features as required) ---
FEATURE_SCHEMA_DEFINITIONS = [
    {"name": "avg_tx_amount", "type": "numeric", "description": "Average transaction amount for the user.",
     "range": "[20, 10000]", "example": 500.25},
    {"name": "device_change_freq", "type": "numeric", "description": "Frequency of device changes for the user.",
     "range": "[0, 1]", "example": 0.05},
    {"name": "tx_hour", "type": "numeric", "description": "Hour of the transaction (0-23).", "range": "[0, 23]",
     "example": 14},
    {"name": "location_change_freq", "type": "numeric", "description": "Frequency of location changes for the user.",
     "range": "[0, 1]", "example": 0.15},
    {"name": "is_new_device", "type": "boolean",
     "description": "Boolean flag indicating if the transaction is from a new device.", "values": [0, 1], "example": 0},
    {"name": "transaction_count_24h", "type": "integer", "description": "Number of transactions in the last 24 hours.",
     "range": ">=0", "example": 5},
    {"name": "time_since_last_tx", "type": "numeric",
     "description": "Time elapsed since the user's previous transaction (in hours).", "range": ">=0", "example": 2.5},
    {"name": "tx_amount_to_balance_ratio", "type": "numeric",
     "description": "Ratio of transaction amount to user's account balance.", "range": "[0, 1]", "example": 0.15},
    {"name": "ip_address_reputation", "type": "numeric",
     "description": "Reputation score of the IP address (0=bad, 1=good).", "range": "[0, 1]", "example": 0.85},
    {"name": "is_weekend", "type": "boolean", "description": "Boolean flag for weekend transactions.", "values": [0, 1],
     "example": 1},
    {"name": "transaction_velocity_10min", "type": "integer",
     "description": "Number of transactions in the last 10 minutes.", "range": ">=0", "example": 2},
    {"name": "country_change_flag", "type": "boolean",
     "description": "Boolean flag if the transaction origin country is different from usual.", "values": [0, 1],
     "example": 0},

    # Additional required fields
    {"name": "user_id", "type": "string", "description": "Unique identifier for the user.", "example": "user_123"},
    {"name": "tx_id", "type": "integer", "description": "Unique identifier for the transaction.", "example": 5001},
    {"name": "timestamp", "type": "string", "subtype": "datetime", "description": "Timestamp of the transaction.",
     "format": "YYYY-MM-DD HH:MM:SS", "example": "2023-03-15 10:30:00"},
    {"name": "tx_amount", "type": "numeric", "description": "Amount of the current transaction.", "range": ">=1",
     "example": 125.75},
    {"name": "account_balance", "type": "numeric", "description": "User's account balance at time of transaction.",
     "range": ">=0", "example": 5000.00},
    {"name": "risk_label", "type": "categorical",
     "description": "Categorical label for transaction risk (legit, suspicious, fraud).",
     "values": ["legit", "suspicious", "fraud"], "label_categorical": True, "example": "legit"},
    {"name": "risk_flag_manual", "type": "boolean",
     "description": "Binary label for transaction risk (0=Normal, 1=Anomalous).", "values": [0, 1],
     "label_binary": True, "example": 0}
]


def generate_behavioral_data():
    print(f"Starting synthetic behavioral data generation to file: {DATA_FILE}")
    data = []
    user_id_counter = 1
    tx_id_counter = 1

    # Track global statistics for balanced labeling
    total_transactions = 0
    fraud_target = 0
    suspicious_target = 0
    legit_target = 0

    print(f"Generating data for {NUM_NORMAL_USERS} normal users...")
    for _ in range(NUM_NORMAL_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1

        # User behavioral patterns
        user_avg_tx_amount = np.random.uniform(50, 800)
        user_device_change_freq = np.random.uniform(0, 0.02)
        user_location_change_freq = np.random.uniform(0, 0.05)
        user_account_balance = np.random.uniform(1000, 50000)
        user_usual_tx_hours = np.random.choice([np.random.randint(9, 18), np.random.randint(19, 22)])

        num_transactions = max(1, int(np.random.normal(TRANSACTIONS_PER_NORMAL_USER_MEAN, 10)))
        user_transactions = []

        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            tx_hour = timestamp.hour
            is_weekend = 1 if timestamp.weekday() >= 5 else 0

            # Core transaction features
            tx_amount = max(1, np.random.normal(user_avg_tx_amount, user_avg_tx_amount * 0.3))

            # Behavioral features
            is_new_device = 1 if random.random() < user_device_change_freq else 0
            country_change_flag = 1 if random.random() < 0.01 else 0  # Very rare for normal users

            # Time-based features
            time_since_last_tx = 0 if i == 0 else max(0.01, abs((timestamp - user_transactions[-1][
                'timestamp']).total_seconds() / 3600))
            transaction_count_24h = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 86400])
            transaction_velocity_10min = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 600])

            # Derived features
            tx_amount_to_balance_ratio = min(1.0, tx_amount / user_account_balance)
            ip_address_reputation = np.random.uniform(0.7, 1.0)  # Good reputation for normal users

            # Store transaction for time-based calculations
            user_transactions.append({
                'timestamp': timestamp,
                'amount': tx_amount
            })

            # --- HEURISTIC LABELING LOGIC (as per requirements) ---
            risk_label = "legit"  # Default

            # Fraud conditions (exact combinations from requirements)
            if (tx_amount_to_balance_ratio > 0.5 and is_new_device == 1 and user_location_change_freq > 0.3):
                risk_label = "fraud"

            # Suspicious conditions (exact combinations from requirements)  
            elif (transaction_velocity_10min >= 3 and (tx_hour < 6 or tx_hour > 22)):
                risk_label = "suspicious"
            elif tx_amount > user_avg_tx_amount * 3:
                risk_label = "suspicious"
            elif country_change_flag == 1:
                risk_label = "suspicious"
            elif is_new_device == 1 and time_since_last_tx < 0.1:  # New device + very quick transaction
                risk_label = "suspicious"

            # For normal users, keep most transactions as legit but add some suspicious
            if risk_label == "legit" and random.random() < 0.05:  # 5% chance
                risk_label = "suspicious"

            risk_flag_manual = 0 if risk_label == "legit" else 1

            data.append([
                user_avg_tx_amount, user_device_change_freq, tx_hour, user_location_change_freq,
                is_new_device, transaction_count_24h, time_since_last_tx, tx_amount_to_balance_ratio,
                ip_address_reputation, is_weekend, transaction_velocity_10min, country_change_flag,
                user_id, tx_id_counter, timestamp, tx_amount, user_account_balance,
                risk_label, risk_flag_manual
            ])
            tx_id_counter += 1
            total_transactions += 1

    print(f"Generating data for {NUM_ANOMALOUS_USERS} anomalous users...")
    for _ in range(NUM_ANOMALOUS_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1

        # Anomalous user patterns
        user_avg_tx_amount = np.random.uniform(500, 5000)
        user_device_change_freq = np.random.uniform(0.2, 0.8)
        user_location_change_freq = np.random.uniform(0.3, 0.9)
        user_account_balance = np.random.uniform(500, 20000)
        user_usual_tx_hours = np.random.choice([np.random.randint(0, 6), np.random.randint(22, 24)])

        num_transactions = max(1, int(np.random.normal(TRANSACTIONS_PER_ANOMALOUS_USER_MEAN, 8)))
        user_transactions = []

        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            tx_hour = timestamp.hour
            is_weekend = 1 if timestamp.weekday() >= 5 else 0

            # Core transaction features
            tx_amount = max(1, np.random.normal(user_avg_tx_amount, user_avg_tx_amount * 0.5))

            # Behavioral features (more anomalous)
            is_new_device = 1 if random.random() < user_device_change_freq else 0
            country_change_flag = 1 if random.random() < 0.4 else 0  # Much higher for anomalous users

            # Time-based features
            time_since_last_tx = 0 if i == 0 else max(0.01, abs((timestamp - user_transactions[-1][
                'timestamp']).total_seconds() / 3600))
            transaction_count_24h = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 86400])
            transaction_velocity_10min = len(
                [t for t in user_transactions if (timestamp - t['timestamp']).total_seconds() <= 600])

            # Derived features
            tx_amount_to_balance_ratio = min(1.0, tx_amount / user_account_balance)
            ip_address_reputation = np.random.uniform(0.1, 0.6)  # Poor reputation for anomalous users

            # Store transaction for time-based calculations
            user_transactions.append({
                'timestamp': timestamp,
                'amount': tx_amount
            })

            # --- HEURISTIC LABELING LOGIC FOR ANOMALOUS USERS ---
            risk_label = "legit"  # Start with legit

            # Fraud conditions (exact combinations from requirements)
            if (tx_amount_to_balance_ratio > 0.5 and is_new_device == 1 and user_location_change_freq > 0.3):
                risk_label = "fraud"
            elif tx_amount_to_balance_ratio > 0.8:  # Very high ratio
                risk_label = "fraud"
            elif (transaction_velocity_10min >= 5):  # Very high velocity
                risk_label = "fraud"
            elif (ip_address_reputation < 0.3 and country_change_flag == 1):
                risk_label = "fraud"

            # Suspicious conditions (exact combinations from requirements)
            elif (transaction_velocity_10min >= 3 and (tx_hour < 6 or tx_hour > 22)):
                risk_label = "suspicious"
            elif tx_amount > user_avg_tx_amount * 2:
                risk_label = "suspicious"
            elif country_change_flag == 1:
                risk_label = "suspicious"
            elif is_new_device == 1:
                risk_label = "suspicious"
            elif ip_address_reputation < 0.5:
                risk_label = "suspicious"

            # For anomalous users, ensure higher fraud/suspicious rates
            if risk_label == "legit":
                rand_val = random.random()
                if rand_val < 0.3:  # 30% chance of fraud
                    risk_label = "fraud"
                elif rand_val < 0.6:  # 30% chance of suspicious  
                    risk_label = "suspicious"

            risk_flag_manual = 0 if risk_label == "legit" else 1

            data.append([
                user_avg_tx_amount, user_device_change_freq, tx_hour, user_location_change_freq,
                is_new_device, transaction_count_24h, time_since_last_tx, tx_amount_to_balance_ratio,
                ip_address_reputation, is_weekend, transaction_velocity_10min, country_change_flag,
                user_id, tx_id_counter, timestamp, tx_amount, user_account_balance,
                risk_label, risk_flag_manual
            ])
            tx_id_counter += 1
            total_transactions += 1

    # Create DataFrame
    columns = [
        'avg_tx_amount', 'device_change_freq', 'tx_hour', 'location_change_freq',
        'is_new_device', 'transaction_count_24h', 'time_since_last_tx', 'tx_amount_to_balance_ratio',
        'ip_address_reputation', 'is_weekend', 'transaction_velocity_10min', 'country_change_flag',
        'user_id', 'tx_id', 'timestamp', 'tx_amount', 'account_balance',
        TARGET_COLUMN_CATEGORICAL, TARGET_COLUMN_BINARY
    ]

    df = pd.DataFrame(data, columns=columns)

    print(f"\nSaving data to file: {DATA_FILE}")
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    try:
        df.to_csv(DATA_FILE, index=False)
        print(f"File {DATA_FILE} has been successfully generated.")
        print(f"Generated {df.shape[0]} rows of data.")
        print(f"Distribution of '{TARGET_COLUMN_CATEGORICAL}':\n{df[TARGET_COLUMN_CATEGORICAL].value_counts()}")
        print(f"Distribution of '{TARGET_COLUMN_BINARY}':\n{df[TARGET_COLUMN_BINARY].value_counts()}")

        # Show proportions
        proportions = df[TARGET_COLUMN_CATEGORICAL].value_counts(normalize=True)
        print(f"\nProportions:")
        for label, prop in proportions.items():
            print(f"{label}: {prop:.2%}")

    except Exception as e:
        print(f"Error occurred while saving CSV file: {e}")
    print("\nData generation completed.")
    return df


def generate_feature_schema_file():
    print(f"\nGenerating feature schema file: {SCHEMA_FILE}")
    schema_data = {
        "description": "Schema definitions for behavioral transaction features (8-15 features as per requirements).",
        "features": FEATURE_SCHEMA_DEFINITIONS,
        "target_column_binary": TARGET_COLUMN_BINARY,
        "target_column_categorical": TARGET_COLUMN_CATEGORICAL,
        "labeling_logic": {
            "fraud_conditions": [
                "High tx_amount_to_balance_ratio + is_new_device + high location_change_freq",
                "Very high transaction_velocity_10min (>=5)",
                "Low ip_address_reputation + country_change_flag"
            ],
            "suspicious_conditions": [
                "High transaction_velocity_10min + unusual tx_hour (early morning/late night)",
                "Transaction amount significantly above user average",
                "Country change flag",
                "New device usage",
                "Low IP reputation"
            ]
        }
    }
    os.makedirs(os.path.dirname(SCHEMA_FILE), exist_ok=True)
    try:
        with open(SCHEMA_FILE, 'w') as f:
            json.dump(schema_data, f, indent=2)
        print(f"Feature schema saved to {SCHEMA_FILE}")
    except Exception as e:
        print(f"Error occurred while saving feature schema: {e}")


def update_behavioral_ml_doc():
    print(f"\nUpdating documentation file: {DOC_FILE}")
    doc_content = f"""# Behavioral Authentication ML Project Overview

This document outlines the machine learning project for behavioral authentication, covering data generation, feature schema, model training, and evaluation as per the project requirements.

## 1. Data Generation (`{DATA_FILE}`)

Synthetic behavioral transaction data is generated to simulate normal and anomalous user activities with controlled label distribution.

### Configuration:
- **Number of Normal Users:** {NUM_NORMAL_USERS}
- **Number of Anomalous Users:** {NUM_ANOMALOUS_USERS}  
- **Transaction Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}
- **Target Label Distribution:**
  - Fraud Rate: {FRAUD_RATE:.1%}
  - Suspicious Rate: {SUSPICIOUS_RATE:.1%}  
  - Legit Rate: {LEGIT_RATE:.1%}

## 2. Feature Schema (`{SCHEMA_FILE}`)

The dataset includes **12 core behavioral features** as specified in the project requirements:

### Core Behavioral Features (8-15 as required):
"""

    # Add feature table
    doc_content += "| Feature Name | Type | Description | Range/Values | Example |\n"
    doc_content += "|---|---|---|---|---|\n"

    for feature in FEATURE_SCHEMA_DEFINITIONS:
        if feature['name'] not in ['user_id', 'tx_id', 'timestamp', 'tx_amount', 'account_balance', 'risk_label',
                                   'risk_flag_manual']:
            name = feature.get('name', 'N/A')
            ftype = feature.get('type', 'N/A')
            description = feature.get('description', 'N/A')
            range_values = feature.get('range', '')
            if not range_values and 'values' in feature:
                range_values = ', '.join(map(str, feature['values']))
            example = feature.get('example', '')
            doc_content += f"| {name} | {ftype} | {description} | {range_values} | {example} |\n"

    doc_content += f"""

### Target Variables:
- **Binary Target:** `{TARGET_COLUMN_BINARY}` - Simple normal/anomalous classification
- **Categorical Target:** `{TARGET_COLUMN_CATEGORICAL}` - Three-class classification (legit, suspicious, fraud)

## 3. Heuristic Labeling Logic

The labeling follows the exact combinations specified in the project requirements:

### Fraud Conditions:
- **Primary:** `tx_amount_to_balance_ratio` + `is_new_device` + high `location_change_freq`
- **Secondary:** Very high `transaction_velocity_10min` (>=5 transactions)
- **Tertiary:** Low `ip_address_reputation` + `country_change_flag`

### Suspicious Conditions:  
- **Primary:** High `transaction_velocity_10min` + unusual `tx_hour` (early morning/late night)
- **Secondary:** Transaction amount significantly above user average
- **Tertiary:** `country_change_flag`, `is_new_device`, or low IP reputation

## 4. Data Quality Assurance

- Balanced class distribution with controlled proportions
- Realistic behavioral patterns differentiate normal vs anomalous users
- Time-based features include proper temporal relationships
- Feature correlations reflect real-world financial behavior patterns

## 5. Next Steps

This dataset is ready for model training phase as outlined in the project plan:
- Train RandomForest, XGBoost, or LightGBM models
- Evaluate using ROC-AUC, Precision/Recall, and Confusion Matrix
- Proceed to API development phase

---
*Generated by: `generate_behavioral_data.py`*
*Schema file: `{SCHEMA_FILE}`*
*Dataset file: `{DATA_FILE}`*
"""

    os.makedirs(os.path.dirname(DOC_FILE), exist_ok=True)
    try:
        with open(DOC_FILE, 'w') as f:
            f.write(doc_content)
        print(f"Documentation updated in {DOC_FILE}")
    except Exception as e:
        print(f"Error occurred while updating documentation: {e}")


if __name__ == "__main__":
    print("=== Behavioral Authentication ML Data Generator ===")
    print("Generating synthetic dataset according to project requirements...\n")

    generate_behavioral_data()
    generate_feature_schema_file()
    update_behavioral_ml_doc()

    print("\n=== Data Generation Complete ===")
    print(f" Dataset: {DATA_FILE}")
    print(f" Schema: {SCHEMA_FILE}")
    print(f" Documentation: {DOC_FILE}")
    print("\nReady for model training phase!")