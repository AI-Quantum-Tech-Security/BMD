import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration class for data generation parameters"""
    data_file: str = 'files/synthetic_behavioral_dataset.csv'
    schema_file: str = 'files/feature_schema.json'
    doc_file: str = 'files/Behavioral_Authentication_ML.md'

    # User configuration
    num_normal_users: int = 1000
    num_anomalous_users: int = 100
    transactions_per_normal_user_mean: int = 50
    transactions_per_anomalous_user_mean: int = 20

    # Date range
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2023, 12, 31)

    # Label distribution
    fraud_rate: float = 0.05
    suspicious_rate: float = 0.15
    legit_rate: float = 0.80

    # Target columns
    target_column_binary: str = 'risk_flag_manual'
    target_column_categorical: str = 'risk_label'


class BehavioralDataGenerator:
    """Enhanced behavioral authentication data generator with comprehensive features"""

    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self.feature_schema = self._define_feature_schema()
        self.user_profiles = {}
        self.transaction_history = {}

    def _define_feature_schema(self) -> List[Dict[str, Any]]:
        """Define comprehensive feature schema for behavioral authentication"""
        return [
            # Transaction amount features
            {"name": "avg_tx_amount", "type": "numeric", "description": "Average transaction amount for the user",
             "range": "[20, 10000]", "example": 500.25, "category": "transaction"},
            {"name": "tx_amount_zscore", "type": "numeric",
             "description": "Z-score of current transaction amount vs user history",
             "range": "[-5, 5]", "example": 1.2, "category": "transaction"},
            {"name": "tx_amount_percentile", "type": "numeric",
             "description": "Percentile of current transaction in user history",
             "range": "[0, 100]", "example": 75.5, "category": "transaction"},

            # Device and location features
            {"name": "device_change_freq", "type": "numeric", "description": "Frequency of device changes for the user",
             "range": "[0, 1]", "example": 0.05, "category": "device"},
            {"name": "is_new_device", "type": "boolean",
             "description": "Boolean flag indicating if transaction is from new device",
             "values": [0, 1], "example": 0, "category": "device"},
            {"name": "device_fingerprint_similarity", "type": "numeric",
             "description": "Similarity to user's typical device fingerprint",
             "range": "[0, 1]", "example": 0.85, "category": "device"},
            {"name": "location_change_freq", "type": "numeric",
             "description": "Frequency of location changes for the user",
             "range": "[0, 1]", "example": 0.15, "category": "location"},
            {"name": "country_change_flag", "type": "boolean",
             "description": "Flag if transaction origin country differs from usual",
             "values": [0, 1], "example": 0, "category": "location"},
            {"name": "distance_from_usual_location", "type": "numeric",
             "description": "Distance from user's usual transaction locations (km)",
             "range": "[0, 20000]", "example": 15.5, "category": "location"},

            # Temporal features
            {"name": "tx_hour", "type": "numeric", "description": "Hour of the transaction (0-23)",
             "range": "[0, 23]", "example": 14, "category": "temporal"},
            {"name": "is_weekend", "type": "boolean", "description": "Boolean flag for weekend transactions",
             "values": [0, 1], "example": 1, "category": "temporal"},
            {"name": "is_holiday", "type": "boolean", "description": "Boolean flag for holiday transactions",
             "values": [0, 1], "example": 0, "category": "temporal"},
            {"name": "time_since_last_tx", "type": "numeric",
             "description": "Time elapsed since user's previous transaction (hours)",
             "range": ">=0", "example": 2.5, "category": "temporal"},
            {"name": "tx_frequency_deviation", "type": "numeric",
             "description": "Deviation from user's typical transaction frequency",
             "range": "[-5, 5]", "example": -0.8, "category": "temporal"},

            # Velocity and volume features
            {"name": "transaction_count_24h", "type": "integer",
             "description": "Number of transactions in last 24 hours",
             "range": ">=0", "example": 5, "category": "velocity"},
            {"name": "transaction_velocity_10min", "type": "integer",
             "description": "Number of transactions in last 10 minutes",
             "range": ">=0", "example": 2, "category": "velocity"},
            {"name": "transaction_volume_24h", "type": "numeric",
             "description": "Total transaction volume in last 24 hours",
             "range": ">=0", "example": 1250.75, "category": "velocity"},
            {"name": "avg_tx_interval", "type": "numeric",
             "description": "Average time interval between user's transactions (hours)",
             "range": ">0", "example": 48.5, "category": "velocity"},

            # Account and risk features
            {"name": "tx_amount_to_balance_ratio", "type": "numeric",
             "description": "Ratio of transaction amount to account balance",
             "range": "[0, 1]", "example": 0.15, "category": "account"},
            {"name": "account_age_days", "type": "integer", "description": "Age of the account in days",
             "range": ">=0", "example": 365, "category": "account"},
            {"name": "ip_address_reputation", "type": "numeric",
             "description": "Reputation score of IP address (0=bad, 1=good)",
             "range": "[0, 1]", "example": 0.85, "category": "network"},
            {"name": "vpn_proxy_flag", "type": "boolean", "description": "Flag indicating VPN or proxy usage",
             "values": [0, 1], "example": 0, "category": "network"},

            # Behavioral consistency features
            {"name": "typing_pattern_similarity", "type": "numeric",
             "description": "Similarity to user's typical typing patterns",
             "range": "[0, 1]", "example": 0.92, "category": "behavior"},
            {"name": "mouse_movement_similarity", "type": "numeric",
             "description": "Similarity to user's typical mouse movement patterns",
             "range": "[0, 1]", "example": 0.88, "category": "behavior"},
            {"name": "session_duration_zscore", "type": "numeric",
             "description": "Z-score of session duration vs user average",
             "range": "[-5, 5]", "example": 0.3, "category": "behavior"},

            # Required system fields
            {"name": "user_id", "type": "string", "description": "Unique identifier for the user",
             "example": "user_123", "category": "system"},
            {"name": "tx_id", "type": "integer", "description": "Unique identifier for the transaction",
             "example": 5001, "category": "system"},
            {"name": "timestamp", "type": "string", "subtype": "datetime",
             "description": "Timestamp of the transaction",
             "format": "YYYY-MM-DD HH:MM:SS", "example": "2023-03-15 10:30:00", "category": "system"},
            {"name": "tx_amount", "type": "numeric", "description": "Amount of the current transaction",
             "range": ">=1", "example": 125.75, "category": "system"},
            {"name": "account_balance", "type": "numeric",
             "description": "User's account balance at time of transaction",
             "range": ">=0", "example": 5000.00, "category": "system"},

            # Target variables
            {"name": "risk_label", "type": "categorical",
             "description": "Categorical risk label (legit, suspicious, fraud)",
             "values": ["legit", "suspicious", "fraud"], "label_categorical": True, "example": "legit",
             "category": "target"},
            {"name": "risk_flag_manual", "type": "boolean", "description": "Binary risk flag (0=Normal, 1=Anomalous)",
             "values": [0, 1], "label_binary": True, "example": 0, "category": "target"}
        ]

    def _create_user_profile(self, user_id: str, is_anomalous: bool = False) -> Dict[str, Any]:
        """Create comprehensive user behavioral profile"""
        if is_anomalous:
            profile = {
                'user_id': user_id,
                'is_anomalous': True,
                'avg_tx_amount': np.random.uniform(500, 5000),
                'tx_amount_std': np.random.uniform(200, 1000),
                'device_change_freq': np.random.uniform(0.2, 0.8),
                'location_change_freq': np.random.uniform(0.3, 0.9),
                'account_balance': np.random.uniform(500, 20000),
                'preferred_hours': list(np.random.choice(range(0, 6), size=2)) + list(
                    np.random.choice(range(22, 24), size=2)),
                'account_age_days': np.random.randint(30, 1000),
                'typical_countries': ['US'] + random.choices(['CN', 'RU', 'NG', 'BR'], k=random.randint(1, 3)),
                'risk_tolerance': np.random.uniform(0.7, 0.95),
                'typing_consistency': np.random.uniform(0.3, 0.7),
                'mouse_consistency': np.random.uniform(0.2, 0.6),
                'session_duration_mean': np.random.uniform(2, 15),
                'vpn_usage_prob': np.random.uniform(0.3, 0.8)
            }
        else:
            profile = {
                'user_id': user_id,
                'is_anomalous': False,
                'avg_tx_amount': np.random.uniform(50, 800),
                'tx_amount_std': np.random.uniform(20, 200),
                'device_change_freq': np.random.uniform(0, 0.02),
                'location_change_freq': np.random.uniform(0, 0.05),
                'account_balance': np.random.uniform(1000, 50000),
                'preferred_hours': list(np.random.choice(range(9, 18), size=3)) + list(
                    np.random.choice(range(19, 22), size=2)),
                'account_age_days': np.random.randint(180, 2000),
                'typical_countries': ['US'],
                'risk_tolerance': np.random.uniform(0.1, 0.3),
                'typing_consistency': np.random.uniform(0.8, 0.98),
                'mouse_consistency': np.random.uniform(0.85, 0.98),
                'session_duration_mean': np.random.uniform(5, 30),
                'vpn_usage_prob': np.random.uniform(0, 0.05)
            }

        self.user_profiles[user_id] = profile
        self.transaction_history[user_id] = []
        return profile

    def _calculate_behavioral_features(self, user_profile: Dict, transaction_data: Dict,
                                       user_history: List[Dict]) -> Dict[str, Any]:
        """Calculate advanced behavioral features based on user profile and history"""
        features = {}

        # Transaction amount features
        if user_history:
            amounts = [tx['tx_amount'] for tx in user_history]
            features['tx_amount_zscore'] = (transaction_data['tx_amount'] - np.mean(amounts)) / (np.std(amounts) + 1e-8)
            features['tx_amount_percentile'] = (sum(1 for amt in amounts if amt <= transaction_data['tx_amount']) / len(
                amounts)) * 100
        else:
            features['tx_amount_zscore'] = 0
            features['tx_amount_percentile'] = 50

        # Device features
        features['device_fingerprint_similarity'] = np.random.uniform(0.7, 1.0) if not user_profile[
            'is_anomalous'] else np.random.uniform(0.2, 0.8)

        # Location features
        features['distance_from_usual_location'] = (
            np.random.exponential(10) if not user_profile['is_anomalous']
            else np.random.exponential(500)
        )

        # Temporal features
        current_hour = transaction_data['timestamp'].hour
        features['tx_frequency_deviation'] = (
                                                     abs(current_hour - np.mean(
                                                         user_profile['preferred_hours'])) / 12 - 0.5
                                             ) * 2

        # Velocity features
        now = transaction_data['timestamp']
        last_24h = [tx for tx in user_history if (now - tx['timestamp']).total_seconds() <= 86400]
        features['transaction_volume_24h'] = sum(tx['tx_amount'] for tx in last_24h)

        if user_history:
            intervals = [(user_history[i]['timestamp'] - user_history[i - 1]['timestamp']).total_seconds() / 3600
                         for i in range(1, len(user_history))]
            features['avg_tx_interval'] = np.mean(intervals) if intervals else 48.0
        else:
            features['avg_tx_interval'] = 48.0

        # Account features
        features['account_age_days'] = user_profile['account_age_days']

        # Network features
        features['vpn_proxy_flag'] = 1 if np.random.random() < user_profile['vpn_usage_prob'] else 0

        # Behavioral consistency features
        features['typing_pattern_similarity'] = np.random.normal(user_profile['typing_consistency'], 0.1)
        features['mouse_movement_similarity'] = np.random.normal(user_profile['mouse_consistency'], 0.1)
        features['session_duration_zscore'] = np.random.normal(0, 1)

        # Ensure all features are within valid ranges
        features['typing_pattern_similarity'] = np.clip(features['typing_pattern_similarity'], 0, 1)
        features['mouse_movement_similarity'] = np.clip(features['mouse_movement_similarity'], 0, 1)

        return features

    def _apply_advanced_labeling_logic(self, features: Dict[str, Any]) -> Tuple[str, int]:
        """Apply sophisticated heuristic labeling logic"""

        # Initialize fraud score
        fraud_score = 0
        suspicious_score = 0

        # High-risk fraud indicators
        if (features.get('tx_amount_to_balance_ratio', 0) > 0.5 and
                features.get('is_new_device', 0) == 1 and
                features.get('location_change_freq', 0) > 0.3):
            fraud_score += 3

        if features.get('transaction_velocity_10min', 0) >= 5:
            fraud_score += 2

        if (features.get('ip_address_reputation', 1) < 0.3 and
                features.get('country_change_flag', 0) == 1):
            fraud_score += 2

        if features.get('distance_from_usual_location', 0) > 1000:
            fraud_score += 1

        if (features.get('typing_pattern_similarity', 1) < 0.5 and
                features.get('mouse_movement_similarity', 1) < 0.5):
            fraud_score += 2

        # Suspicious indicators
        if (features.get('transaction_velocity_10min', 0) >= 3 and
                (features.get('tx_hour', 12) < 6 or features.get('tx_hour', 12) > 22)):
            suspicious_score += 2

        if abs(features.get('tx_amount_zscore', 0)) > 2:
            suspicious_score += 1

        if features.get('country_change_flag', 0) == 1:
            suspicious_score += 1

        if features.get('is_new_device', 0) == 1:
            suspicious_score += 1

        if features.get('vpn_proxy_flag', 0) == 1:
            suspicious_score += 1

        if features.get('session_duration_zscore', 0) > 2:
            suspicious_score += 1

        # Decision logic
        if fraud_score >= 3:
            return "fraud", 1
        elif fraud_score >= 2 or suspicious_score >= 3:
            return "suspicious", 1
        elif suspicious_score >= 1:
            # Add some randomness for edge cases
            if np.random.random() < 0.3:
                return "suspicious", 1

        return "legit", 0

    def generate_comprehensive_dataset(self) -> pd.DataFrame:
        """Generate comprehensive behavioral authentication dataset"""
        logger.info("Starting comprehensive behavioral data generation")

        data = []
        tx_id_counter = 1

        # Generate normal users
        logger.info(f"Generating data for {self.config.num_normal_users} normal users")
        for i in range(self.config.num_normal_users):
            user_id = f'user_{i + 1:06d}'
            user_profile = self._create_user_profile(user_id, is_anomalous=False)

            num_transactions = max(1, int(np.random.normal(
                self.config.transactions_per_normal_user_mean,
                self.config.transactions_per_normal_user_mean * 0.2
            )))

            for j in range(num_transactions):
                transaction_data = self._generate_single_transaction(
                    user_profile, tx_id_counter, j == 0
                )

                behavioral_features = self._calculate_behavioral_features(
                    user_profile, transaction_data, self.transaction_history[user_id]
                )

                # Combine all features
                combined_features = {**transaction_data, **behavioral_features}

                # Apply labeling logic
                risk_label, risk_flag = self._apply_advanced_labeling_logic(combined_features)
                combined_features['risk_label'] = risk_label
                combined_features['risk_flag_manual'] = risk_flag

                data.append(combined_features)

                # Update transaction history
                self.transaction_history[user_id].append(transaction_data)
                tx_id_counter += 1

        # Generate anomalous users
        logger.info(f"Generating data for {self.config.num_anomalous_users} anomalous users")
        for i in range(self.config.num_anomalous_users):
            user_id = f'anom_user_{i + 1:06d}'
            user_profile = self._create_user_profile(user_id, is_anomalous=True)

            num_transactions = max(1, int(np.random.normal(
                self.config.transactions_per_anomalous_user_mean,
                self.config.transactions_per_anomalous_user_mean * 0.3
            )))

            for j in range(num_transactions):
                transaction_data = self._generate_single_transaction(
                    user_profile, tx_id_counter, j == 0
                )

                behavioral_features = self._calculate_behavioral_features(
                    user_profile, transaction_data, self.transaction_history[user_id]
                )

                # Combine all features
                combined_features = {**transaction_data, **behavioral_features}

                # Apply labeling logic with higher anomaly bias
                risk_label, risk_flag = self._apply_advanced_labeling_logic(combined_features)

                # Increase anomaly rate for anomalous users
                if risk_label == "legit" and np.random.random() < 0.4:
                    risk_label = "suspicious" if np.random.random() < 0.7 else "fraud"
                    risk_flag = 1

                combined_features['risk_label'] = risk_label
                combined_features['risk_flag_manual'] = risk_flag

                data.append(combined_features)

                # Update transaction history
                self.transaction_history[user_id].append(transaction_data)
                tx_id_counter += 1

        # Create DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} total transactions")

        return df

    def _generate_single_transaction(self, user_profile: Dict, tx_id: int, is_first: bool) -> Dict[str, Any]:
        """Generate a single transaction with realistic features"""

        # Generate timestamp
        timestamp = self.config.start_date + timedelta(
            seconds=random.randint(0, int((self.config.end_date - self.config.start_date).total_seconds()))
        )

        # Basic transaction features
        tx_amount = max(1, np.random.normal(
            user_profile['avg_tx_amount'],
            user_profile['tx_amount_std']
        ))

        transaction = {
            'user_id': user_profile['user_id'],
            'tx_id': tx_id,
            'timestamp': timestamp,
            'tx_amount': tx_amount,
            'account_balance': user_profile['account_balance'],
            'avg_tx_amount': user_profile['avg_tx_amount'],
            'device_change_freq': user_profile['device_change_freq'],
            'location_change_freq': user_profile['location_change_freq'],
            'tx_hour': timestamp.hour,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_holiday': 1 if self._is_holiday(timestamp) else 0,
            'is_new_device': 1 if np.random.random() < user_profile['device_change_freq'] else 0,
            'country_change_flag': 1 if np.random.random() < (0.4 if user_profile['is_anomalous'] else 0.01) else 0,
            'tx_amount_to_balance_ratio': min(1.0, tx_amount / user_profile['account_balance']),
            'ip_address_reputation': (
                np.random.uniform(0.1, 0.6) if user_profile['is_anomalous']
                else np.random.uniform(0.7, 1.0)
            )
        }

        # Calculate time-based features
        user_history = self.transaction_history.get(user_profile['user_id'], [])

        if is_first or not user_history:
            transaction['time_since_last_tx'] = 0
            transaction['transaction_count_24h'] = 0
            transaction['transaction_velocity_10min'] = 0
        else:
            last_tx = user_history[-1]
            transaction['time_since_last_tx'] = abs(
                (timestamp - last_tx['timestamp']).total_seconds() / 3600
            )

            # Count transactions in time windows
            transaction['transaction_count_24h'] = len([
                tx for tx in user_history
                if (timestamp - tx['timestamp']).total_seconds() <= 86400
            ])

            transaction['transaction_velocity_10min'] = len([
                tx for tx in user_history
                if (timestamp - tx['timestamp']).total_seconds() <= 600
            ])

        return transaction

    def _is_holiday(self, date: datetime) -> bool:
        """Simple holiday detection (can be expanded)"""
        # Simple implementation - major US holidays
        major_holidays = [
            (1, 1),  # New Year's Day
            (7, 4),  # Independence Day
            (12, 25),  # Christmas
        ]
        return (date.month, date.day) in major_holidays

    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save dataset with validation and statistics"""
        os.makedirs(os.path.dirname(self.config.data_file), exist_ok=True)

        try:
            df.to_csv(self.config.data_file, index=False)
            logger.info(f"Dataset saved successfully to {self.config.data_file}")

            # Print statistics
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Risk label distribution:\n{df['risk_label'].value_counts()}")
            logger.info(f"Risk flag distribution:\n{df['risk_flag_manual'].value_counts()}")

            # Calculate proportions
            proportions = df['risk_label'].value_counts(normalize=True)
            logger.info("Risk label proportions:")
            for label, prop in proportions.items():
                logger.info(f"  {label}: {prop:.2%}")

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

    def generate_data_quality_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive data quality report"""
        logger.info("Generating data quality report")

        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

        report = {
            'dataset_info': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum()
            },
            'feature_statistics': {},
            'label_distribution': df['risk_label'].value_counts().to_dict(),
            'class_balance': df['risk_label'].value_counts(normalize=True).to_dict()
        }

        # Feature statistics
        for col in numeric_cols:
            if col not in ['tx_id', 'timestamp']:
                report['feature_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'missing_count': int(df[col].isnull().sum())
                }

        # Save report
        report_file = self.config.data_file.replace('.csv', '_quality_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Data quality report saved to {report_file}")

    def save_feature_schema(self) -> None:
        """Save enhanced feature schema"""
        schema_data = {
            "description": "Enhanced schema for behavioral authentication features",
            "version": "2.0",
            "features": self.feature_schema,
            "target_column_binary": self.config.target_column_binary,
            "target_column_categorical": self.config.target_column_categorical,
            "feature_categories": {
                "transaction": "Features related to transaction amounts and patterns",
                "device": "Features related to device usage and changes",
                "location": "Features related to geographic patterns",
                "temporal": "Features related to timing patterns",
                "velocity": "Features related to transaction frequency and volume",
                "account": "Features related to account characteristics",
                "network": "Features related to network and IP characteristics",
                "behavior": "Features related to user behavioral patterns",
                "system": "System and identification features",
                "target": "Target variables for prediction"
            },
            "labeling_logic": {
                "fraud_conditions": [
                    "High transaction-to-balance ratio + new device + high location change frequency",
                    "Very high transaction velocity (>=5 in 10 minutes)",
                    "Low IP reputation + country change flag",
                    "Large distance from usual location + device inconsistency",
                    "Low typing and mouse pattern similarity"
                ],
                "suspicious_conditions": [
                    "High transaction velocity + unusual hours",
                    "High Z-score for transaction amount",
                    "Country change flag",
                    "New device usage",
                    "VPN/Proxy usage",
                    "Unusual session duration"
                ]
            }
        }

        os.makedirs(os.path.dirname(self.config.schema_file), exist_ok=True)

        try:
            with open(self.config.schema_file, 'w') as f:
                json.dump(schema_data, f, indent=2)
            logger.info(f"Feature schema saved to {self.config.schema_file}")
        except Exception as e:
            logger.error(f"Error saving feature schema: {e}")
            raise


def main():
    """Main execution function"""
    print("=== Enhanced Behavioral Authentication ML Data Generator ===")

    # Initialize configuration and generator
    config = DataGenerationConfig()
    generator = BehavioralDataGenerator(config)

    # Generate dataset
    df = generator.generate_comprehensive_dataset()

    # Save dataset
    generator.save_dataset(df)

    # Generate reports
    generator.generate_data_quality_report(df)
    generator.save_feature_schema()

    print("\n=== Generation Complete ===")
    print(f"Dataset: {config.data_file}")
    print(f"Schema: {config.schema_file}")
    print(f"Quality Report: {config.data_file.replace('.csv', '_quality_report.json')}")
    print("\nReady for model training phase!")


if __name__ == "__main__":
    main()