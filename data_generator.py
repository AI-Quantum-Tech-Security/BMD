import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
import uuid
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Configure logging
os.makedirs('files/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('files/logs/data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration class for data generation parameters"""
    # File paths
    data_file: str = 'files/synthetic_behavioral_dataset.csv'
    schema_file: str = 'files/json/feature_schema.json'
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

    # Behavioral patterns
    min_typing_speed: int = 30
    max_typing_speed: int = 120
    min_session_duration: int = 2
    max_session_duration: int = 60
    location_radius: float = 100.0

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization"""
        if self.num_normal_users <= 0 or self.num_anomalous_users <= 0:
            raise ValueError("Number of users must be positive")

        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")

        if not (0 <= self.fraud_rate <= 1 and 0 <= self.suspicious_rate <= 1):
            raise ValueError("Rates must be between 0 and 1")

        total_rate = self.fraud_rate + self.suspicious_rate + self.legit_rate
        if abs(total_rate - 1.0) > 0.01:
            logger.warning(f"Label rates sum to {total_rate}, not 1.0")


class BehavioralDataGenerator:
    """Enhanced behavioral authentication data generator with comprehensive features"""

    def __init__(self, config: DataGenerationConfig) -> None:
        """Initialize the data generator with configuration validation"""
        self.config = config
        self.feature_schema = self._define_feature_schema()
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(f"Initialized generator with {config.num_normal_users} normal "
                    f"and {config.num_anomalous_users} anomalous users")

    def _define_feature_schema(self) -> List[Dict[str, Any]]:
        """Define comprehensive feature schema for behavioral authentication"""
        return [
            # Transaction amount features
            {"name": "avg_tx_amount", "type": "numeric",
             "description": "Average transaction amount for the user",
             "range": "[20, 10000]", "example": 500.25, "category": "transaction"},

            {"name": "tx_amount_zscore", "type": "numeric",
             "description": "Z-score of current transaction amount vs user history",
             "range": "[-5, 5]", "example": 1.2, "category": "transaction"},

            {"name": "tx_amount_percentile", "type": "numeric",
             "description": "Percentile of current transaction in user history",
             "range": "[0, 100]", "example": 75.5, "category": "transaction"},

            # Device and location features
            {"name": "device_change_freq", "type": "numeric",
             "description": "Frequency of device changes for the user",
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
            {"name": "tx_hour", "type": "numeric",
             "description": "Hour of the transaction (0-23)",
             "range": "[0, 23]", "example": 14, "category": "temporal"},

            {"name": "is_weekend", "type": "boolean",
             "description": "Boolean flag for weekend transactions",
             "values": [0, 1], "example": 1, "category": "temporal"},

            {"name": "is_holiday", "type": "boolean",
             "description": "Boolean flag for holiday transactions",
             "values": [0, 1], "example": 0, "category": "temporal"},

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

            # Account and risk features
            {"name": "tx_amount_to_balance_ratio", "type": "numeric",
             "description": "Ratio of transaction amount to account balance",
             "range": "[0, 1]", "example": 0.15, "category": "account"},

            {"name": "account_age_days", "type": "integer",
             "description": "Age of the account in days",
             "range": ">=0", "example": 365, "category": "account"},

            {"name": "ip_address_reputation", "type": "numeric",
             "description": "Reputation score of IP address (0=bad, 1=good)",
             "range": "[0, 1]", "example": 0.85, "category": "network"},

            {"name": "vpn_proxy_flag", "type": "boolean",
             "description": "Flag indicating VPN or proxy usage",
             "values": [0, 1], "example": 0, "category": "network"},

            # Behavioral consistency features
            {"name": "typing_pattern_similarity", "type": "numeric",
             "description": "Similarity to user's typical typing patterns",
             "range": "[0, 1]", "example": 0.92, "category": "behavior"},

            {"name": "mouse_movement_similarity", "type": "numeric",
             "description": "Similarity to user's typical mouse movement patterns",
             "range": "[0, 1]", "example": 0.88, "category": "behavior"},

            # System fields
            {"name": "user_id", "type": "string",
             "description": "Unique identifier for the user",
             "example": "user_123", "category": "system"},

            {"name": "tx_id", "type": "integer",
             "description": "Unique identifier for the transaction",
             "example": 5001, "category": "system"},

            {"name": "timestamp", "type": "datetime",
             "description": "Timestamp of the transaction",
             "format": "YYYY-MM-DD HH:MM:SS", "example": "2023-03-15 10:30:00",
             "category": "system"},

            {"name": "tx_amount", "type": "numeric",
             "description": "Amount of the current transaction",
             "range": ">=1", "example": 125.75, "category": "system"},

            {"name": "account_balance", "type": "numeric",
             "description": "User's account balance at time of transaction",
             "range": ">=0", "example": 5000.00, "category": "system"},

            # Target variables
            {"name": "risk_label", "type": "categorical",
             "description": "Categorical risk label (legit, suspicious, fraud)",
             "values": ["legit", "suspicious", "fraud"], "label_categorical": True,
             "example": "legit", "category": "target"},

            {"name": "risk_flag_manual", "type": "boolean",
             "description": "Binary risk flag (0=Normal, 1=Anomalous)",
             "values": [0, 1], "label_binary": True, "example": 0, "category": "target"}
        ]

    def _create_user_profile(self, user_id: str, is_anomalous: bool = False) -> Dict[str, Any]:
        """Create enhanced user behavioral profile with realistic patterns"""
        profile = {
            'user_id': user_id,
            'is_anomalous': is_anomalous,
            'activity_level': np.random.choice(['low', 'medium', 'high']),
            'risk_tolerance': np.random.uniform(0.1, 0.3) if not is_anomalous
            else np.random.uniform(0.7, 0.95)
        }

        # Set temporal patterns based on activity level
        activity_patterns = {
            'low': {
                'tx_per_day': np.random.uniform(0.5, 2),
                'active_hours': np.random.choice(range(9, 18), size=3)
            },
            'medium': {
                'tx_per_day': np.random.uniform(2, 5),
                'active_hours': list(np.random.choice(range(9, 18), size=4)) +
                                list(np.random.choice(range(19, 22), size=2))
            },
            'high': {
                'tx_per_day': np.random.uniform(5, 10),
                'active_hours': list(np.random.choice(range(8, 22), size=8))
            }
        }

        patterns = activity_patterns[profile['activity_level']]

        # Transaction patterns
        profile.update({
            'avg_tx_amount': np.random.uniform(50, 800) if not is_anomalous
            else np.random.uniform(500, 5000),
            'tx_amount_std': np.random.uniform(20, 200) if not is_anomalous
            else np.random.uniform(200, 1000),
            'tx_per_day': patterns['tx_per_day'],
            'preferred_hours': patterns['active_hours'],
            'weekend_activity_factor': np.random.uniform(0.3, 0.7) if not is_anomalous
            else np.random.uniform(0.6, 1.2)
        })

        # Device and location patterns
        profile.update({
            'device_change_freq': np.random.uniform(0, 0.02) if not is_anomalous
            else np.random.uniform(0.2, 0.8),
            'location_change_freq': np.random.uniform(0, 0.05) if not is_anomalous
            else np.random.uniform(0.3, 0.9),
            'typical_locations': self._generate_typical_locations(is_anomalous),
            'typical_devices': self._generate_device_profiles(is_anomalous)
        })

        # Behavioral patterns
        profile.update({
            'typing_consistency': np.random.uniform(0.8, 0.98) if not is_anomalous
            else np.random.uniform(0.3, 0.7),
            'mouse_consistency': np.random.uniform(0.85, 0.98) if not is_anomalous
            else np.random.uniform(0.2, 0.6),
            'session_patterns': self._generate_session_patterns(is_anomalous),
            'account_balance': np.random.uniform(1000, 50000) if not is_anomalous
            else np.random.uniform(500, 20000),
            'account_age_days': np.random.randint(180, 2000) if not is_anomalous
            else np.random.randint(30, 1000),
            'vpn_usage_prob': np.random.uniform(0, 0.05) if not is_anomalous
            else np.random.uniform(0.3, 0.8)
        })

        self.user_profiles[user_id] = profile
        self.transaction_history[user_id] = []
        return profile

    def _generate_typical_locations(self, is_anomalous: bool) -> List[Dict[str, Any]]:
        """Generate realistic location patterns"""
        num_locations = np.random.randint(1, 3) if not is_anomalous else np.random.randint(3, 8)
        locations = []

        for _ in range(num_locations):
            locations.append({
                'lat': np.random.uniform(-90, 90),
                'lon': np.random.uniform(-180, 180),
                'frequency': np.random.uniform(0.1, 1.0),
                'typical_hours': list(np.random.choice(range(24), size=np.random.randint(4, 12)))
            })
        return locations

    def _generate_device_profiles(self, is_anomalous: bool) -> List[Dict[str, Any]]:
        """Generate realistic device usage patterns"""
        num_devices = np.random.randint(1, 3) if not is_anomalous else np.random.randint(3, 6)
        devices = []

        for _ in range(num_devices):
            devices.append({
                'fingerprint': f"device_{uuid.uuid4().hex[:8]}",
                'usage_frequency': np.random.uniform(0.1, 1.0),
                'typical_hours': list(np.random.choice(range(24), size=np.random.randint(4, 12))),
                'typical_locations': np.random.randint(1, 3)
            })
        return devices

    def _generate_session_patterns(self, is_anomalous: bool) -> Dict[str, Any]:
        """Generate realistic session behavior patterns"""
        return {
            'avg_session_duration': np.random.uniform(5, 30) if not is_anomalous
            else np.random.uniform(2, 15),
            'typing_speed_range': (
                np.random.uniform(40, 100) if not is_anomalous else np.random.uniform(20, 150),
                np.random.uniform(60, 120) if not is_anomalous else np.random.uniform(30, 200)
            ),
            'mouse_movement_patterns': {
                'avg_speed': np.random.uniform(100, 300),
                'avg_acceleration': np.random.uniform(50, 150),
                'consistency': np.random.uniform(0.8, 0.95) if not is_anomalous
                else np.random.uniform(0.3, 0.7)
            }
        }

    def _calculate_behavioral_features(self, user_profile: Dict[str, Any],
                                       transaction_data: Dict[str, Any],
                                       user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhanced behavioral features"""
        features = {}

        # Time-based features
        hour = transaction_data['timestamp'].hour
        is_preferred_hour = hour in user_profile['preferred_hours']

        # Transaction amount features
        if user_history:
            amounts = [tx['tx_amount'] for tx in user_history]
            mean_amount = np.mean(amounts)
            std_amount = np.std(amounts) + 1e-8  # Avoid division by zero

            features['tx_amount_zscore'] = (transaction_data['tx_amount'] - mean_amount) / std_amount
            features['tx_amount_percentile'] = (
                                                       sum(1 for amt in amounts if
                                                           amt <= transaction_data['tx_amount']) / len(amounts)
                                               ) * 100
        else:
            features['tx_amount_zscore'] = 0
            features['tx_amount_percentile'] = 50

        # Session and behavioral features
        typing_base = user_profile['typing_consistency']
        mouse_base = user_profile['mouse_consistency']

        # Add time-of-day effect on behavioral patterns
        if not is_preferred_hour:
            typing_base *= np.random.uniform(0.8, 0.95)
            mouse_base *= np.random.uniform(0.8, 0.95)

        features['typing_pattern_similarity'] = np.clip(
            np.random.normal(typing_base, 0.1), 0, 1
        )
        features['mouse_movement_similarity'] = np.clip(
            np.random.normal(mouse_base, 0.1), 0, 1
        )

        # Location and device features
        features['device_fingerprint_similarity'] = (
            np.random.uniform(0.8, 1.0) if not transaction_data.get('is_new_device', 0)
            else np.random.uniform(0.2, 0.6)
        )

        # Calculate location-based features
        features['distance_from_usual_location'] = (
            np.random.exponential(10) if not user_profile['is_anomalous']
            else np.random.exponential(500)
        )

        # Network features
        features['vpn_proxy_flag'] = 1 if np.random.random() < user_profile['vpn_usage_prob'] else 0
        features['ip_address_reputation'] = (
            np.random.uniform(0.7, 1.0) if not user_profile['is_anomalous']
            else np.random.uniform(0.1, 0.6)
        )

        # Velocity features
        now = transaction_data['timestamp']
        last_24h = [tx for tx in user_history if (now - tx['timestamp']).total_seconds() <= 86400]
        last_10min = [tx for tx in user_history if (now - tx['timestamp']).total_seconds() <= 600]

        features['transaction_count_24h'] = len(last_24h)
        features['transaction_volume_24h'] = sum(tx['tx_amount'] for tx in last_24h)
        features['transaction_velocity_10min'] = len(last_10min)

        return features

    def _apply_advanced_labeling_logic(self, features: Dict[str, Any]) -> Tuple[str, int]:
        """Enhanced risk scoring with sophisticated rules"""
        risk_score = 0.0

        # High-risk indicators
        if features.get('tx_amount_to_balance_ratio', 0) > 0.5:
            risk_score += 0.3
            if features.get('is_new_device', 0) == 1:
                risk_score += 0.2

        if features.get('transaction_velocity_10min', 0) >= 5:
            risk_score += 0.4

        # Location and device risk
        if features.get('distance_from_usual_location', 0) > 1000:
            risk_score += 0.2
            if features.get('is_new_device', 0) == 1:
                risk_score += 0.2

        # Behavioral patterns risk
        behavioral_score = (
                                   (1 - features.get('typing_pattern_similarity', 1)) +
                                   (1 - features.get('mouse_movement_similarity', 1))
                           ) / 2
        risk_score += behavioral_score * 0.3

        # Network risk
        if features.get('vpn_proxy_flag', 0) == 1:
            risk_score += 0.1
            if features.get('ip_address_reputation', 1) < 0.3:
                risk_score += 0.2

        # Final risk determination
        if risk_score > 0.7:
            return "fraud", 1
        elif risk_score > 0.4:
            return "suspicious", 1
        else:
            return "legit", 0

    def generate_comprehensive_dataset(self) -> pd.DataFrame:
        """Generate comprehensive behavioral authentication dataset with progress tracking"""
        logger.info("Starting comprehensive behavioral data generation")

        data = []
        tx_id_counter = 1
        total_users = self.config.num_normal_users + self.config.num_anomalous_users

        # Generate normal users
        logger.info(f"Generating data for {self.config.num_normal_users} normal users")
        for i in range(self.config.num_normal_users):
            if i % 100 == 0:  # Progress indicator
                logger.info(f"Progress: {i}/{self.config.num_normal_users} normal users processed")

            user_id = f'user_{i + 1:06d}'
            user_profile = self._create_user_profile(user_id, is_anomalous=False)

            num_transactions = max(1, int(np.random.normal(
                self.config.transactions_per_normal_user_mean,
                self.config.transactions_per_normal_user_mean * 0.2
            )))

            self._generate_user_transactions(user_profile, num_transactions, tx_id_counter, data)
            tx_id_counter += num_transactions

        # Generate anomalous users
        logger.info(f"Generating data for {self.config.num_anomalous_users} anomalous users")
        for i in range(self.config.num_anomalous_users):
            if i % 25 == 0:  # Progress indicator
                logger.info(f"Progress: {i}/{self.config.num_anomalous_users} anomalous users processed")

            user_id = f'anom_user_{i + 1:06d}'
            user_profile = self._create_user_profile(user_id, is_anomalous=True)

            num_transactions = max(1, int(np.random.normal(
                self.config.transactions_per_anomalous_user_mean,
                self.config.transactions_per_anomalous_user_mean * 0.3
            )))

            self._generate_user_transactions(user_profile, num_transactions, tx_id_counter, data)
            tx_id_counter += num_transactions

        # Create DataFrame
        logger.info("Creating DataFrame from generated data")
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} total transactions from {total_users} users")

        return df

    def _generate_user_transactions(self, user_profile: Dict[str, Any], num_transactions: int,
                                    start_tx_id: int, data: List[Dict[str, Any]]) -> None:
        """Generate transactions for a specific user"""
        for j in range(num_transactions):
            transaction_data = self._generate_single_transaction(
                user_profile, start_tx_id + j, j == 0
            )

            behavioral_features = self._calculate_behavioral_features(
                user_profile, transaction_data, self.transaction_history[user_profile['user_id']]
            )

            # Combine features
            combined_features = {**transaction_data, **behavioral_features}

            # Apply labeling logic
            risk_label, risk_flag = self._apply_advanced_labeling_logic(combined_features)

            # Increase anomaly rate for anomalous users
            if user_profile['is_anomalous'] and risk_label == "legit" and np.random.random() < 0.4:
                risk_label = "suspicious" if np.random.random() < 0.7 else "fraud"
                risk_flag = 1

            combined_features['risk_label'] = risk_label
            combined_features['risk_flag_manual'] = risk_flag

            data.append(combined_features)
            self.transaction_history[user_profile['user_id']].append(transaction_data)

    def _generate_single_transaction(self, user_profile: Dict[str, Any], tx_id: int,
                                     is_first: bool) -> Dict[str, Any]:
        """Generate a single transaction with enhanced realistic features"""
        # Generate timestamp with temporal patterns
        timestamp = self._generate_realistic_timestamp(user_profile)

        # Basic transaction features
        tx_amount = self._generate_transaction_amount(user_profile, timestamp)

        transaction = {
            'user_id': user_profile['user_id'],
            'tx_id': tx_id,
            'timestamp': timestamp,
            'tx_amount': tx_amount,
            'account_balance': user_profile['account_balance'],
            'tx_hour': timestamp.hour,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_holiday': 1 if self._is_holiday(timestamp) else 0,
            'is_new_device': 1 if np.random.random() < user_profile['device_change_freq'] else 0,
            'country_change_flag': 1 if np.random.random() < (
                0.4 if user_profile['is_anomalous'] else 0.01) else 0,
            'tx_amount_to_balance_ratio': min(1.0, tx_amount / user_profile['account_balance'])
        }

        return transaction

    def _generate_realistic_timestamp(self, user_profile: Dict[str, Any]) -> datetime:
        """Generate timestamp based on user's temporal patterns"""
        max_attempts = 100  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            timestamp = self.config.start_date + timedelta(
                seconds=random.randint(
                    0,
                    int((self.config.end_date - self.config.start_date).total_seconds())
                )
            )

            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5

            # Check if timestamp matches user's patterns
            if hour in user_profile['preferred_hours']:
                return timestamp
            elif is_weekend and np.random.random() < user_profile['weekend_activity_factor']:
                return timestamp
            elif np.random.random() < 0.2:  # Allow some transactions outside normal patterns
                return timestamp

            attempts += 1

        # Fallback to ensure we always return a timestamp
        return self.config.start_date + timedelta(
            seconds=random.randint(
                0, int((self.config.end_date - self.config.start_date).total_seconds())
            )
        )

    def _generate_transaction_amount(self, user_profile: Dict[str, Any],
                                     timestamp: datetime) -> float:
        """Generate realistic transaction amount based on user patterns and time"""
        base_amount = np.random.normal(user_profile['avg_tx_amount'], user_profile['tx_amount_std'])

        # Apply temporal factors
        if timestamp.weekday() >= 5:  # Weekend
            base_amount *= user_profile['weekend_activity_factor']

        if timestamp.hour not in user_profile['preferred_hours']:
            base_amount *= np.random.uniform(0.5, 1.5)

        return max(1.0, base_amount)

    def _is_holiday(self, date: datetime) -> bool:
        """Enhanced holiday detection"""
        major_holidays = [
            (1, 1),  # New Year's Day
            (12, 25),  # Christmas
            (12, 26),  # Boxing Day
            (7, 4),  # Independence Day (US)
            (11, 11),  # Veterans Day (US)
            (5, 1),  # Labor Day (Many countries)
        ]
        return (date.month, date.day) in major_holidays

    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save dataset with enhanced validation and statistics"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.data_file), exist_ok=True)

            # Validate data
            self._validate_dataset(df)

            # Save data
            df.to_csv(self.config.data_file, index=False)
            logger.info(f"Dataset saved successfully to {self.config.data_file}")

            # Generate and save statistics
            self._generate_dataset_statistics(df)

            # Save feature schema
            self._save_feature_schema()

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate dataset quality"""
        logger.info("Validating dataset quality...")

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")

        # Check value ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].min() < 0 and col not in ['tx_amount_zscore']:
                logger.warning(f"Negative values found in column {col}")

        # Check class balance
        class_balance = df['risk_label'].value_counts(normalize=True)
        logger.info(f"Class balance:\n{class_balance}")

        # Validate data types
        expected_dtypes = {
            'user_id': 'object',
            'tx_id': 'int64',
            'tx_amount': 'float64',
            'risk_label': 'object',
            'risk_flag_manual': 'int64'
        }

        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns and df[col].dtype != expected_dtype:
                logger.warning(f"Column {col} has dtype {df[col].dtype}, expected {expected_dtype}")

    def _generate_dataset_statistics(self, df: pd.DataFrame) -> None:
        """Generate comprehensive dataset statistics"""
        logger.info("Generating dataset statistics...")

        stats = {
            'generation_info': {
                'generated_at': datetime.now().isoformat(),
                'config': {
                    'num_normal_users': self.config.num_normal_users,
                    'num_anomalous_users': self.config.num_anomalous_users,
                    'date_range': f"{self.config.start_date} to {self.config.end_date}"
                }
            },
            'basic_stats': {
                'num_transactions': len(df),
                'num_users': df['user_id'].nunique(),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'class_distribution': df['risk_label'].value_counts().to_dict(),
                'avg_transaction_amount': float(df['tx_amount'].mean()),
                'med_transaction_amount': float(df['tx_amount'].median()),
                'total_volume': float(df['tx_amount'].sum())
            },
            'temporal_patterns': {
                'hourly_distribution': df.groupby(df['timestamp'].dt.hour)['tx_id'].count().to_dict(),
                'weekday_distribution': df.groupby(df['timestamp'].dt.dayofweek)['tx_id'].count().to_dict(),
                'monthly_distribution': df.groupby(df['timestamp'].dt.month)['tx_id'].count().to_dict()
            },
            'risk_patterns': {
                'risk_by_hour': df.groupby(['tx_hour', 'risk_label']).size().unstack(fill_value=0).to_dict(),
                'risk_by_amount_range': df.groupby(['risk_label'])['tx_amount'].agg(
                    ['mean', 'std', 'min', 'max']).to_dict()
            },
            'feature_stats': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
                for col in df.select_dtypes(include=[np.number]).columns
            }
        }

        # Save statistics
        stats_file = os.path.join('files', 'json',
                                  os.path.basename(self.config.data_file).replace('.csv', '_statistics.json'))
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Dataset statistics saved to {stats_file}")

    def _save_feature_schema(self) -> None:
        """Save feature schema to JSON file"""
        schema_data = {
            'schema_version': '1.0',
            'description': 'Behavioral Authentication ML Feature Schema',
            'generated_at': datetime.now().isoformat(),
            'features': self.feature_schema,
            'target_variables': {
                'binary': self.config.target_column_binary,
                'categorical': self.config.target_column_categorical
            }
        }

        os.makedirs(os.path.dirname(self.config.schema_file), exist_ok=True)
        with open(self.config.schema_file, 'w') as f:
            json.dump(schema_data, f, indent=2)
        logger.info(f"Feature schema saved to {self.config.schema_file}")


def main() -> None:
    """Main execution function with error handling"""
    try:
        print("=== Enhanced Behavioral Authentication ML Data Generator ===")
        print(f"Generated by: olafcio42")
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Initialize configuration and generator
        config = DataGenerationConfig()
        generator = BehavioralDataGenerator(config)

        # Generate dataset
        logger.info(" Starting dataset generation...")
        df = generator.generate_comprehensive_dataset()

        # Save dataset and generate reports
        generator.save_dataset(df)

        print("\n=== Generation Complete ===")
        print(f"Dataset: {config.data_file}")
        print(f"Schema: {config.schema_file}")
        print(f"Statistics: {config.data_file.replace('.csv', '_statistics.json')}")
        print(f"Total transactions: {len(df):,}")
        print(f"Total users: {df['user_id'].nunique():,}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()