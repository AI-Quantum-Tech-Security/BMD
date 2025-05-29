# Behavioral Authentication ML Project Overview

This document outlines the machine learning project for behavioral authentication, covering data generation, feature schema, model training, and evaluation.

## 1. Data Generation (`synthetic_behavioral_data.csv`)
Synthetic behavioral transaction data is generated to simulate normal and anomalous user activities.
-**Number of Normal Users:** 1000
-**Number of Anomalous Users:** 100
-**Transaction Period:** 2023-01-01 to 2023-12-31

## 2. Feature Schema (`feature_schema.json`)
The following table describes the features included in the dataset.

| Feature Name | Type | Subtype | Description | Range/Values | Example |
|---|---|---|---|---|---|
| session_duration | numeric |  | Duration of user session in seconds. | >=10 | 150.5 |
| login_time_pattern | string | time | Pattern of user login time (HH:MM). |  | 14:35 |
| avg_tx_amount | numeric |  | Average transaction amount for the user (based on generation pattern). | [20, 10000] | 500.25 |
| geo_distance_delta | numeric |  | Geographical distance change from previous transaction/login location. | >=0 | 75.8 |
| user_id | string |  | Unique identifier for the user. |  | user_123 |
| tx_id | integer |  | Unique identifier for the transaction. |  | 5001 |
| timestamp | string | datetime | Timestamp of the transaction. |  | 2023-03-15 10:30:00 |
| tx_amount | numeric |  | Amount of the current transaction. | >=1 | 125.75 |
| currency | categorical |  | Currency of the transaction. | PLN, EUR, USD, GBP, JPY | USD |
| tx_type | categorical |  | Type of transaction. | purchase, transfer, withdrawal, online_payment, international_transfer | purchase |
| merchant_id | categorical |  | Identifier of the merchant involved in the transaction. |  | merchant_42 |
| tx_location | categorical |  | Location of the transaction. |  | loc_25 |
| device_id | categorical |  | Identifier of the device used for the transaction. |  | dev_5 |
| ip_address | string | IP_address | IP address used for the transaction. |  | 192.168.1.10 |
| is_vpn | boolean |  | Flag indicating if a VPN was detected (0=No, 1=Yes). | 0, 1 | 0 |
| avg_tx_amount_user | numeric |  | Average transaction amount for the specific user (pattern). | [20, 10000] | 480.12 |
| std_tx_amount_user | numeric |  | Standard deviation of transaction amount for the specific user (pattern). | >=0 | 55.6 |
| avg_tx_hour_user | numeric |  | Average hour of the day for transactions for the specific user (pattern). | [0, 23] | 14.5 |
| device_change_freq | numeric |  | Frequency of device changes for the user. | [0, 1] | 0.05 |
| location_change_freq | numeric |  | Frequency of location changes for the user. | [0, 1] | 0.15 |
| txs_last_24h | integer |  | Number of transactions in the last 24 hours for the user. | >=0 | 5 |
| txs_last_7d | integer |  | Number of transactions in the last 7 days for the user. | >=0 | 20 |
| has_recent_password_reset | boolean |  | Flag indicating if the user had a recent password reset (0=No, 1=Yes). | 0, 1 | 0 |
| is_new_device | boolean |  | Flag indicating if a new device was used (0=No, 1=Yes). | 0, 1 | 0 |
| tx_hour | numeric |  | Hour of the current transaction (0-23). | [0, 23] | 12 |
| risk_flag_manual | boolean |  | Manual label for transaction risk (0=Normal, 1=Anomalous/Fraud). This is the target variable. | 0, 1 |  |
| anomaly_score_baseline | numeric |  | Baseline anomaly score from a theoretical previous system. Not used as a feature for this model. | [0, 1] | 0.15 |
| country_mismatch | boolean |  | Flag indicating if transaction country mismatches user's usual country (0=No, 1=Yes). | 0, 1 | 0 |
| is_weekend | boolean |  | Flag indicating if the transaction occurred on a weekend (0=No, 1=Yes). | 0, 1 | 1 |
| ip_risk_score | numeric |  | Risk score associated with the IP address. | [0, 1] | 0.08 |

-**Target Column:** `risk_flag_manual` - used for labeling transactions as normal or anomalous.

## 3. Model Training & Evaluation (Refer to `train_evaluate_model.py`)

## 4. API & Deployment (Future Work / Separate Deliverable)

## Next Steps & Considerations
-**Advanced Preprocessing:** Explore robust imputation and high-cardinality categorical feature handling.
-**Feature Engineering:** Create new features from existing ones.
-**Class Imbalance:** Use techniques like SMOTE or `class_weight`.
-**Model Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV.
-**Cross-validation:** Implement for robust evaluation.
-**Alternative Models:** Experiment with XGBoost or LightGBM.
-**Deployment:** Prepare model for API deployment consistency.
