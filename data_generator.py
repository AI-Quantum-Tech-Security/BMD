import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json

#---Global Configuration---
DATA_FILE = 'synthetic_behavioral_data.csv' # Changed filename
SCHEMA_FILE = 'feature_schema.json'
DOC_FILE = 'Behavioral_Authentication_ML.md'

#---Data Generation Configuration---
NUM_NORMAL_USERS = 1000
NUM_ANOMALOUS_USERS = 100
TRANSACTIONS_PER_NORMAL_USER_MEAN = 50
TRANSACTIONS_PER_ANOMALOUS_USER_MEAN = 20 # Can be adjusted
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)
TARGET_COLUMN = 'risk_flag_manual'

#---Feature Schema Definition (unchanged, as the features themselves are the same)---
FEATURE_SCHEMA_DEFINITIONS = [
    {"name": "session_duration", "type": "numeric", "description": "Duration of user session in seconds.", "range": ">=10", "example": 150.5},
    {"name": "login_time_pattern", "type": "string", "subtype": "time", "description": "Pattern of user login time (HH:MM).", "format": "HH:MM", "example": "14:35"},
    {"name": "avg_tx_amount", "type": "numeric", "description": "Average transaction amount for the user (based on generation pattern).", "range": "[20, 10000]", "example": 500.25},
    {"name": "geo_distance_delta", "type": "numeric", "description": "Geographical distance change from previous transaction/login location.", "range": ">=0", "example": 75.8},
    {"name": "user_id", "type": "string", "description": "Unique identifier for the user.", "example": "user_123"},
    {"name": "tx_id", "type": "integer", "description": "Unique identifier for the transaction.", "example": 5001},
    {"name": "timestamp", "type": "string", "subtype": "datetime", "description": "Timestamp of the transaction.", "format": "YYYY-MM-DD HH:MM:SS", "example": "2023-03-15 10:30:00"},
    {"name": "tx_amount", "type": "numeric", "description": "Amount of the current transaction.", "range": ">=1", "example": 125.75},
    {"name": "currency", "type": "categorical", "description": "Currency of the transaction.", "values": ["PLN", "EUR", "USD", "GBP", "JPY"], "example": "USD"},
    {"name": "tx_type", "type": "categorical", "description": "Type of transaction.", "values": ["purchase", "transfer", "withdrawal", "online_payment", "international_transfer"], "example": "purchase"},
    {"name": "merchant_id", "type": "categorical", "description": "Identifier of the merchant involved in the transaction.", "example": "merchant_42"},
    {"name": "tx_location", "type": "categorical", "description": "Location of the transaction.", "example": "loc_25"},
    {"name": "device_id", "type": "categorical", "description": "Identifier of the device used for the transaction.", "example": "dev_5"},
    {"name": "ip_address", "type": "string", "subtype": "IP_address", "description": "IP address used for the transaction.", "example": "192.168.1.10"},
    {"name": "is_vpn", "type": "boolean", "description": "Flag indicating if a VPN was detected (0=No, 1=Yes).", "values": [0, 1], "example": 0},
    {"name": "avg_tx_amount_user", "type": "numeric", "description": "Average transaction amount for the specific user (pattern).", "range": "[20, 10000]", "example": 480.12},
    {"name": "std_tx_amount_user", "type": "numeric", "description": "Standard deviation of transaction amount for the specific user (pattern).", "range": ">=0", "example": 55.6},
    {"name": "avg_tx_hour_user", "type": "numeric", "description": "Average hour of the day for transactions for the specific user (pattern).", "range": "[0, 23]", "example": 14.5},
    {"name": "device_change_freq", "type": "numeric", "description": "Frequency of device changes for the user.", "range": "[0, 1]", "example": 0.05},
    {"name": "location_change_freq", "type": "numeric", "description": "Frequency of location changes for the user.", "range": "[0, 1]", "example": 0.15},
    {"name": "txs_last_24h", "type": "integer", "description": "Number of transactions in the last 24 hours for the user.", "range": ">=0", "example": 5},
    {"name": "txs_last_7d", "type": "integer", "description": "Number of transactions in the last 7 days for the user.", "range": ">=0", "example": 20},
    {"name": "has_recent_password_reset", "type": "boolean", "description": "Flag indicating if the user had a recent password reset (0=No, 1=Yes).", "values": [0, 1], "example": 0},
    {"name": "is_new_device", "type": "boolean", "description": "Flag indicating if a new device was used (0=No, 1=Yes).", "values": [0, 1], "example": 0},
    {"name": "tx_hour", "type": "numeric", "description": "Hour of the current transaction (0-23).", "range": "[0, 23]", "example": 12},
    {"name": "risk_flag_manual", "type": "boolean", "description": "Manual label for transaction risk (0=Normal, 1=Anomalous/Fraud). This is the target variable.", "values": [0, 1], "label": True},
    {"name": "anomaly_score_baseline", "type": "numeric", "description": "Baseline anomaly score from a theoretical previous system. Not used as a feature for this model.", "range": "[0, 1]", "example": 0.15},
    {"name": "country_mismatch", "type": "boolean", "description": "Flag indicating if transaction country mismatches user's usual country (0=No, 1=Yes).", "values": [0, 1], "example": 0},
    {"name": "is_weekend", "type": "boolean", "description": "Flag indicating if the transaction occurred on a weekend (0=No, 1=Yes).", "values": [0, 1], "example": 1},
    {"name": "ip_risk_score", "type": "numeric", "description": "Risk score associated with the IP address.", "range": "[0, 1]", "example": 0.08},
]


def generate_behavioral_data():
    print(f"Starting synthetic behavioral data generation to file: {DATA_FILE}")
    data = []
    user_id_counter = 1
    tx_id_counter = 1

    # Define common ranges for categorical features
    common_merchants = [f'merchant_{i}' for i in range(1, 100)]
    common_locations = [f'loc_{i}' for i in range(1, 50)]
    common_devices = [f'dev_{i}' for i in range(1, 20)]
    common_ips = [f'192.168.1.{i}' for i in range(1, 200)] # Smaller range for common IPs

    # Define less common/anomalous ranges for categorical features
    anomalous_merchants = [f'merchant_{i}' for i in range(50, 150)] # Overlap with common
    anomalous_locations = [f'loc_{i}' for i in range(30, 80)] # Overlap with common
    anomalous_devices = [f'dev_{i}' for i in range(10, 30)] # Overlap with common
    anomalous_ips = [f'10.0.0.{i}' for i in range(1, 100)] + [f'192.168.1.{i}' for i in range(150, 255)] # Mix of new and "risky" common

    print(f"Generating data for {NUM_NORMAL_USERS} normal users...")
    for _ in range(NUM_NORMAL_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1
        # Normal user patterns - more consistent, lower risk
        avg_tx_amount_user_pattern = np.random.uniform(50, 800)
        std_tx_amount_user_pattern = avg_tx_amount_user_pattern * np.random.uniform(0.05, 0.2) # Smaller std deviation
        avg_tx_hour_user_pattern = np.random.randint(9, 20) # Normal business hours
        device_change_freq_base = np.random.uniform(0, 0.02) # Low frequency
        location_change_freq_base = np.random.uniform(0, 0.05) # Low frequency

        # User-specific base risk factors
        user_has_recent_password_reset = 0 # Mostly no for normal
        user_is_new_device = 0 # Mostly no for normal
        user_ip_risk_score_base = np.random.uniform(0.01, 0.1)
        user_country_mismatch_base = 0
        user_is_vpn_base = 0

        num_transactions = max(1, int(np.random.normal(TRANSACTIONS_PER_NORMAL_USER_MEAN, 10)))
        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            tx_hour = timestamp.hour
            is_weekend = 1 if timestamp.weekday() >= 5 else 0

            # Transaction amounts slightly more varied for normal users
            tx_amount = max(1, np.random.normal(avg_tx_amount_user_pattern, std_tx_amount_user_pattern * np.random.uniform(0.8, 1.2)))

            session_duration = max(30, int(np.random.normal(180, 90)))
            geo_distance_delta = max(0, np.random.normal(10, 20)) # Small distance changes

            login_time_pattern = f"{np.random.randint(max(0, avg_tx_hour_user_pattern - 2), min(23, avg_tx_hour_user_pattern + 2)):02d}:{np.random.randint(0,59):02d}"

            txs_last_24h = int(np.random.normal(num_transactions / ((END_DATE - START_DATE).days / 365 * 24) * random.uniform(0.8, 1.2), 2))
            txs_last_7d = int(np.random.normal(num_transactions / ((END_DATE - START_DATE).days / 365 * 7) * random.uniform(0.8, 1.2), 5))

            currency = random.choice(['PLN', 'EUR', 'USD'])
            tx_type = random.choice(['purchase', 'transfer', 'withdrawal', 'online_payment'])

            # Categorical features mostly from common pools, but some occasional "anomalous" ones
            merchant_id = random.choices(common_merchants, weights=[0.95]*len(common_merchants), k=1)[0] if random.random() < 0.9 else random.choice(anomalous_merchants)
            tx_location = random.choices(common_locations, weights=[0.95]*len(common_locations), k=1)[0] if random.random() < 0.9 else random.choice(anomalous_locations)
            device_id = random.choices(common_devices, weights=[0.95]*len(common_devices), k=1)[0] if random.random() < 0.9 else random.choice(anomalous_devices)
            ip_address = random.choices(common_ips, weights=[0.95]*len(common_ips), k=1)[0] if random.random() < 0.9 else random.choice(anomalous_ips)

            # Introduce some noise for binary flags
            is_vpn = 1 if random.random() < 0.02 else 0 # Very rarely for normal users
            has_recent_password_reset = 1 if random.random() < 0.01 else 0
            is_new_device = 1 if random.random() < 0.03 else 0
            country_mismatch = 1 if random.random() < 0.01 else 0
            ip_risk_score = max(0.01, min(0.99, user_ip_risk_score_base * np.random.uniform(0.8, 1.2) + random.uniform(-0.01, 0.01)))

            anomaly_score_baseline = np.random.uniform(0, 0.3) # Baseline can be a bit higher for normal

            data.append([
                session_duration, login_time_pattern, avg_tx_amount_user_pattern, geo_distance_delta,
                user_id, tx_id_counter, timestamp, tx_amount, currency, tx_type, merchant_id,
                tx_location, device_id, ip_address, is_vpn, avg_tx_amount_user_pattern, std_tx_amount_user_pattern,
                avg_tx_hour_user_pattern, device_change_freq_base, location_change_freq_base, txs_last_24h, txs_last_7d,
                has_recent_password_reset, is_new_device, tx_hour, 0, anomaly_score_baseline,
                country_mismatch, is_weekend, ip_risk_score
            ])
            tx_id_counter += 1

    print(f"Generating data for {NUM_ANOMALOUS_USERS} anomalous users...")
    for _ in range(NUM_ANOMALOUS_USERS):
        user_id = f'user_{user_id_counter}'
        user_id_counter += 1
        # Anomalous user patterns - more extreme, but with some overlap with normal
        avg_tx_amount_user_pattern = np.random.uniform(500, 5000) # Still some overlap with normal max
        std_tx_amount_user_pattern = avg_tx_amount_user_pattern * np.random.uniform(0.3, 1.0) # Larger std deviation
        avg_tx_hour_user_pattern = np.random.choice([random.randint(0, 8), random.randint(20, 23)]) # Out-of-hours, but not always
        device_change_freq_base = np.random.uniform(0.1, 0.8) # Higher frequency, but not always 1.0
        location_change_freq_base = np.random.uniform(0.2, 0.9) # Higher frequency, but not always 1.0

        # User-specific base risk factors (more likely to be risky)
        user_has_recent_password_reset = random.choice([0, 1]) # 50/50 for anomalous user
        user_is_new_device = random.choice([0, 1]) # 50/50 for anomalous user
        user_ip_risk_score_base = np.random.uniform(0.2, 0.9) # Higher general risk
        user_country_mismatch_base = random.choice([0, 1])
        user_is_vpn_base = random.choice([0, 1])

        num_transactions = max(1, int(np.random.normal(TRANSACTIONS_PER_ANOMALOUS_USER_MEAN, 15)))
        for i in range(num_transactions):
            timestamp = START_DATE + timedelta(seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds())))
            tx_hour = timestamp.hour
            is_weekend = 1 if timestamp.weekday() >= 5 else 0

            # Transaction amounts for anomalous users - higher, but some overlap
            tx_amount = max(1, np.random.normal(avg_tx_amount_user_pattern, std_tx_amount_user_pattern * np.random.uniform(0.8, 1.5)))

            session_duration = max(5, int(np.random.normal(90, 60))) # Shorter sessions, more varied
            geo_distance_delta = max(0, np.random.normal(100, 200)) # Larger distances, with some smaller ones

            login_time_pattern = f"{np.random.randint(max(0, avg_tx_hour_user_pattern - 3), min(23, avg_tx_hour_user_pattern + 3)):02d}:{np.random.randint(0,59):02d}"

            txs_last_24h = int(np.random.normal(num_transactions / ((END_DATE - START_DATE).days / 365 * 24) * random.uniform(1.0, 3.0), 5)) # More frequent
            txs_last_7d = int(np.random.normal(num_transactions / ((END_DATE - START_DATE).days / 365 * 7) * random.uniform(1.0, 2.0), 10)) # More frequent

            currency = random.choice(['PLN', 'EUR', 'USD', 'GBP', 'JPY']) # Wider range
            tx_type = random.choice(['purchase', 'transfer', 'withdrawal', 'online_payment', 'international_transfer'])

            # Categorical features more likely from anomalous pools, but some from common
            merchant_id = random.choices(anomalous_merchants, weights=[0.8]*len(anomalous_merchants), k=1)[0] if random.random() < 0.8 else random.choice(common_merchants)
            tx_location = random.choices(anomalous_locations, weights=[0.8]*len(anomalous_locations), k=1)[0] if random.random() < 0.8 else random.choice(common_locations)
            device_id = random.choices(anomalous_devices, weights=[0.8]*len(anomalous_devices), k=1)[0] if random.random() < 0.8 else random.choice(common_devices)
            ip_address = random.choices(anomalous_ips, weights=[0.8]*len(anomalous_ips), k=1)[0] if random.random() < 0.8 else random.choice(common_ips)

            # Binary flags are more likely to be 1 for anomalous users
            is_vpn = 1 if random.random() < 0.6 else 0
            has_recent_password_reset = 1 if random.random() < 0.3 else 0 # Still not 100%
            is_new_device = 1 if random.random() < 0.4 else 0 # Still not 100%
            country_mismatch = 1 if random.random() < 0.5 else 0
            ip_risk_score = max(0.01, min(0.99, user_ip_risk_score_base * np.random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)))

            anomaly_score_baseline = np.random.uniform(0.4, 1.0) # Higher baseline for anomalous

            data.append([
                session_duration, login_time_pattern, avg_tx_amount_user_pattern, geo_distance_delta,
                user_id, tx_id_counter, timestamp, tx_amount, currency, tx_type, merchant_id,
                tx_location, device_id, ip_address, is_vpn, avg_tx_amount_user_pattern, std_tx_amount_user_pattern,
                avg_tx_hour_user_pattern, device_change_freq_base, location_change_freq_base, txs_last_24h, txs_last_7d,
                has_recent_password_reset, is_new_device, tx_hour, 1, anomaly_score_baseline,
                country_mismatch, is_weekend, ip_risk_score
            ])
            tx_id_counter += 1

    generated_df_columns = [
        'session_duration', 'login_time_pattern', 'avg_tx_amount', 'geo_distance_delta',
        'user_id', 'tx_id', 'timestamp', 'tx_amount', 'currency', 'tx_type', 'merchant_id',
        'tx_location', 'device_id', 'ip_address', 'is_vpn', 'avg_tx_amount_user', 'std_tx_amount_user',
        'avg_tx_hour_user', 'device_change_freq', 'location_change_freq', 'txs_last_24h', 'txs_last_7d',
        'has_recent_password_reset', 'is_new_device', 'tx_hour', 'risk_flag_manual',
        'anomaly_score_baseline', 'country_mismatch', 'is_weekend', 'ip_risk_score'
    ]

    df = pd.DataFrame(data, columns=generated_df_columns)
    print(f"\nSaving data to file: {DATA_FILE}")
    try:
        df.to_csv(DATA_FILE, index=False)
        print(f"File {DATA_FILE} has been successfully generated.")
        print(f"Generated {df.shape[0]} rows of data.")
    except Exception as e:
        print(f"Error occurred while saving CSV file: {e}")
    print("\nData generation completed.")
    return df

def generate_feature_schema_file():
    print(f"\nGenerating feature schema file: {SCHEMA_FILE}")
    schema_data = {
        "description": "Schema definitions for behavioral transaction features.",
        "features": FEATURE_SCHEMA_DEFINITIONS,
        "target_column": TARGET_COLUMN
    }
    try:
        with open(SCHEMA_FILE, 'w') as f:
            json.dump(schema_data, f, indent=2)
        print(f"Feature schema saved to {SCHEMA_FILE}")
    except Exception as e:
        print(f"Error occurred while saving feature schema: {e}")

def update_behavioral_ml_doc():
    print(f"\nUpdating overall documentation file: {DOC_FILE}")
    doc_content = f"# Behavioral Authentication ML Project Overview\n\n"
    doc_content += f"This document outlines the machine learning project for behavioral authentication, covering data generation, feature schema, model training, and evaluation.\n\n"

    doc_content += f"## 1. Data Generation (`{DATA_FILE}`)\n"
    doc_content += f"Synthetic behavioral transaction data is generated to simulate normal and anomalous user activities.\n"
    doc_content += f"-**Number of Normal Users:** {NUM_NORMAL_USERS}\n"
    doc_content += f"-**Number of Anomalous Users:** {NUM_ANOMALOUS_USERS}\n"
    doc_content += f"-**Transaction Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}\n\n"

    doc_content += f"## 2. Feature Schema (`{SCHEMA_FILE}`)\n"
    doc_content += f"The following table describes the features included in the dataset.\n\n"
    doc_content += f"| Feature Name | Type | Subtype | Description | Range/Values | Example |\n"
    doc_content += f"|---|---|---|---|---|---|\n"

    for feature in FEATURE_SCHEMA_DEFINITIONS:
        name = feature.get('name', 'N/A')
        ftype = feature.get('type', 'N/A')
        subtype = feature.get('subtype', '')
        description = feature.get('description', 'N/A')

        range_values = feature.get('range', '')
        if not range_values and 'values' in feature:
            range_values = ', '.join(map(str, feature['values']))

        example = feature.get('example', '')

        doc_content += f"| {name} | {ftype} | {subtype} | {description} | {range_values} | {example} |\n"
    doc_content += "\n"
    doc_content += f"-**Target Column:** `{TARGET_COLUMN}` - used for labeling transactions as normal or anomalous.\n\n"

    doc_content += f"## 3. Model Training & Evaluation (Refer to `train_evaluate_model.py`)\n\n"
    doc_content += f"## 4. API & Deployment (Future Work / Separate Deliverable)\n\n"

    doc_content += f"## Next Steps & Considerations\n"
    doc_content += f"-**Advanced Preprocessing:** Explore robust imputation and high-cardinality categorical feature handling.\n"
    doc_content += f"-**Feature Engineering:** Create new features from existing ones.\n"
    doc_content += f"-**Class Imbalance:** Use techniques like SMOTE or `class_weight`.\n"
    doc_content += f"-**Model Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV.\n"
    doc_content += f"-**Cross-validation:** Implement for robust evaluation.\n"
    doc_content += f"-**Alternative Models:** Experiment with XGBoost or LightGBM.\n"
    doc_content += f"-**Deployment:** Prepare model for API deployment consistency.\n"

    try:
        with open(DOC_FILE, 'w') as f:
            f.write(doc_content)
        print(f"Overall documentation updated in {DOC_FILE}")
    except Exception as e:
        print(f"Error occurred while updating documentation: {e}")

if __name__ == "__main__":
    generate_behavioral_data()
    generate_feature_schema_file()
    update_behavioral_ml_doc()
    print("\nData generation script finished.")