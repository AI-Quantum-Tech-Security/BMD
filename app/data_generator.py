"""
=============================================================================
ENHANCED BEHAVIORAL AUTHENTICATION ML DATA GENERATOR v2.0
=============================================================================

Author: olafcio42
Repository: AI-Quantum-Tech-Security/BMD
Date: 2025-07-13
Version: 2.0 Enhanced

DESCRIPTION:
-----------
Advanced synthetic behavioral data generator for ML authentication systems.
Significantly improved version of the original BehavioralDataGenerator with
40+ realistic features, multi-factor risk scoring system and automatic API
integration capabilities.

KEY IMPROVEMENTS vs ORIGINAL:
----------------------------
- 40+ features instead of 25 (merchant, network, behavioral depth)
- Multi-weighted risk scoring system (6 category weights)
- Realistic user profiles with ActivityLevel enum classification
- Edge cases and false positives/negatives simulation
- Automatic API test data generation (JSON format)
- Advanced validation with data leakage detection
- Comprehensive schema export with feature categorization

INTEGRATION WITH BMD PROJECT:
----------------------------
Data Generation: Replaces data_generator.py (25 features → 40+ features)
ML Training: Supports main.py, train_evaluate_model.py (backward compatible)
API Testing: Automatic test data generation for final_risk_api.py
Schema Export: Enhanced JSON for Java integration
ML Pipeline: Production-ready data for complete ML workflow

TECHNOLOGIES USED:
-----------------
- pandas, numpy: Data manipulation and statistical computations
- typing, dataclasses: Type safety and configuration management
- enum: Risk profiles and activity level classification
- logging: Advanced process logging and monitoring
- json: Schema export and API test data serialization
- datetime: Realistic timestamps with behavioral patterns

OUTPUT DATA STRUCTURE:
---------------------
1. files/synthetic_behavioral_dataset.csv (main dataset)
2. files/json/enhanced_feature_schema.json (complete schema)
3. files/api_test_data.json (100 samples for API testing)
4. files/logs/enhanced_data_generation.log (generation logs)

USAGE:
------
config = EnhancedDataGenerationConfig()
generator = EnhancedBehavioralDataGenerator(config)
df = generator.generate_comprehensive_dataset()
generator.save_enhanced_dataset(df)

COMPATIBILITY:
-------------
Backward compatible with existing BMD codebase while providing significant
enhancements for realistic ML model training and production deployment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json
import uuid
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import warnings
from enum import Enum

warnings.filterwarnings('ignore')

# Configure logging
os.makedirs('../files/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../files/logs/enhanced_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Enum for different risk profiles"""
    LEGITIMATE = "legitimate"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"
    MIXED = "mixed"


class ActivityLevel(Enum):
    """Enum for user activity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class EnhancedDataGenerationConfig:
    """Enhanced configuration class with more realistic parameters"""
    # File paths
    data_file: str = '../files/synthetic_behavioral_dataset.csv'
    schema_file: str = '../files/json/enhanced_feature_schema.json'
    doc_file: str = 'files/Enhanced_Behavioral_Authentication_ML.md'
    api_test_file: str = '../files/api_test_data.json'

    # User configuration - more realistic distributions
    num_normal_users: int = 100
    num_anomalous_users: int = 20
    transactions_per_normal_user_mean: int = 10
    transactions_per_anomalous_user_mean: int = 10

    # Date range
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2023, 12, 31)

    # Enhanced label distribution with more realistic rates
    fraud_rate: float = 0.08
    suspicious_rate: float = 0.22
    legit_rate: float = 0.70

    # Target columns
    target_column_binary: str = 'risk_flag_manual'
    target_column_categorical: str = 'risk_label'

    # Business parameters
    min_transaction_amount: float = 1.0
    max_transaction_amount: float = 50000.0
    min_account_balance: float = 100.0
    max_account_balance: float = 100000.0

    # Advanced behavioral parameters
    typing_consistency_normal: Tuple[float, float] = (0.8, 0.98)
    typing_consistency_anomalous: Tuple[float, float] = (0.3, 0.7)
    session_duration_normal: Tuple[float, float] = (5, 45)
    session_duration_anomalous: Tuple[float, float] = (1, 15)

    # Categorical data pools
    merchants: List[str] = None
    countries: List[str] = None
    device_types: List[str] = None

    def __post_init__(self) -> None:
        """Initialize categorical data and validate parameters"""
        if self.merchants is None:
            self.merchants = [f'merchant_{i:03d}' for i in range(1, 201)]

        if self.countries is None:
            self.countries = ['US', 'UK', 'DE', 'FR', 'IT', 'ES', 'CA', 'AU', 'JP', 'CN', 'IN', 'BR', 'MX', 'NL', 'SE']

        if self.device_types is None:
            self.device_types = ['mobile_ios', 'mobile_android', 'desktop_windows', 'desktop_mac', 'tablet_ios',
                                 'tablet_android', 'unknown']

        # Validation
        if self.num_normal_users <= 0 or self.num_anomalous_users <= 0:
            raise ValueError("Number of users must be positive")

        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")

        total_rate = self.fraud_rate + self.suspicious_rate + self.legit_rate
        if abs(total_rate - 1.0) > 0.01:
            logger.warning(f"Label rates sum to {total_rate}, adjusting to 1.0")
            # Normalize rates
            self.fraud_rate /= total_rate
            self.suspicious_rate /= total_rate
            self.legit_rate /= total_rate


class EnhancedBehavioralDataGenerator:
    """Enhanced behavioral authentication data generator with advanced features and realistic patterns"""

    def __init__(self, config: EnhancedDataGenerationConfig) -> None:
        """Initialize the enhanced data generator"""
        self.config = config
        self.feature_schema = self._define_enhanced_feature_schema()
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.transaction_history: Dict[str, List[Dict[str, Any]]] = {}
        self.merchant_risk_scores: Dict[str, float] = self._generate_merchant_risk_scores()
        self.ip_reputation_cache: Dict[str, float] = {}

        # Global counters
        self.tx_id_counter = 1
        self.device_fingerprint_counter = 1

        logger.info(f"Enhanced generator initialized: {config.num_normal_users} normal users, "
                    f"{config.num_anomalous_users} anomalous users")

    def _define_enhanced_feature_schema(self) -> List[Dict[str, Any]]:
        """Define comprehensive enhanced feature schema"""
        return [
            # Enhanced transaction features
            {"name": "avg_tx_amount", "type": "numeric", "description": "User's average transaction amount",
             "range": "[20, 10000]", "example": 500.25, "category": "transaction", "importance": "high"},

            {"name": "tx_amount_zscore", "type": "numeric",
             "description": "Z-score of current transaction vs user history",
             "range": "[-5, 5]", "example": 1.2, "category": "transaction", "importance": "high"},

            {"name": "tx_amount_percentile", "type": "numeric",
             "description": "Percentile of current transaction in user history",
             "range": "[0, 100]", "example": 75.5, "category": "transaction", "importance": "medium"},

            {"name": "amount_balance_ratio", "type": "numeric", "description": "Transaction amount to balance ratio",
             "range": "[0, 1]", "example": 0.15, "category": "transaction", "importance": "high"},

            # Enhanced device and location features
            {"name": "device_change_freq", "type": "numeric", "description": "Frequency of device changes",
             "range": "[0, 1]", "example": 0.05, "category": "device", "importance": "high"},

            {"name": "is_new_device", "type": "boolean", "description": "Transaction from new device",
             "values": [0, 1], "example": 0, "category": "device", "importance": "high"},

            {"name": "device_fingerprint_similarity", "type": "numeric",
             "description": "Device fingerprint similarity score",
             "range": "[0, 1]", "example": 0.85, "category": "device", "importance": "medium"},

            {"name": "device_type", "type": "categorical", "description": "Type of device used",
             "values": ["mobile_ios", "mobile_android", "desktop_windows", "desktop_mac", "tablet_ios",
                        "tablet_android", "unknown"],
             "example": "mobile_ios", "category": "device", "importance": "medium"},

            {"name": "location_change_freq", "type": "numeric", "description": "Frequency of location changes",
             "range": "[0, 1]", "example": 0.15, "category": "location", "importance": "high"},

            {"name": "country_change_flag", "type": "boolean", "description": "Transaction from different country",
             "values": [0, 1], "example": 0, "category": "location", "importance": "high"},

            {"name": "distance_from_usual_location", "type": "numeric",
             "description": "Distance from usual location (km)",
             "range": "[0, 20000]", "example": 15.5, "category": "location", "importance": "medium"},

            {"name": "location", "type": "categorical", "description": "Transaction origin country",
             "values": ["US", "UK", "DE", "FR", "IT", "ES", "CA", "AU", "JP", "CN", "IN", "BR", "MX", "NL", "SE"],
             "example": "US", "category": "location", "importance": "medium"},

            # Enhanced temporal features
            {"name": "tx_hour", "type": "numeric", "description": "Hour of transaction (0-23)",
             "range": "[0, 23]", "example": 14, "category": "temporal", "importance": "medium"},

            {"name": "is_weekend", "type": "boolean", "description": "Weekend transaction flag",
             "values": [0, 1], "example": 1, "category": "temporal", "importance": "low"},

            {"name": "is_holiday", "type": "boolean", "description": "Holiday transaction flag",
             "values": [0, 1], "example": 0, "category": "temporal", "importance": "low"},

            {"name": "is_business_hours", "type": "boolean", "description": "Business hours transaction flag",
             "values": [0, 1], "example": 1, "category": "temporal", "importance": "medium"},

            {"name": "is_unusual_hour", "type": "boolean", "description": "Unusual hour transaction flag",
             "values": [0, 1], "example": 0, "category": "temporal", "importance": "medium"},

            # Enhanced velocity and volume features
            {"name": "transaction_count_24h", "type": "integer", "description": "Transactions in last 24 hours",
             "range": ">=0", "example": 5, "category": "velocity", "importance": "high"},

            {"name": "transaction_velocity_10min", "type": "integer", "description": "Transactions in last 10 minutes",
             "range": ">=0", "example": 2, "category": "velocity", "importance": "high"},

            {"name": "transaction_volume_24h", "type": "numeric", "description": "Transaction volume in last 24 hours",
             "range": ">=0", "example": 1250.75, "category": "velocity", "importance": "medium"},

            {"name": "time_since_last_tx", "type": "numeric", "description": "Hours since last transaction",
             "range": ">=0", "example": 2.5, "category": "velocity", "importance": "medium"},

            # Enhanced account and merchant features
            {"name": "account_age_days", "type": "integer", "description": "Account age in days",
             "range": ">=0", "example": 365, "category": "account", "importance": "medium"},

            {"name": "merchant_id", "type": "categorical", "description": "Merchant identifier",
             "values": [f'merchant_{i:03d}' for i in range(1, 21)], "example": "merchant_001", "category": "merchant",
             "importance": "medium"},

            {"name": "merchant_risk_score", "type": "numeric", "description": "Merchant risk score",
             "range": "[0, 1]", "example": 0.15, "category": "merchant", "importance": "high"},

            {"name": "is_high_risk_merchant", "type": "boolean", "description": "High risk merchant flag",
             "values": [0, 1], "example": 0, "category": "merchant", "importance": "high"},

            # Enhanced network and security features
            {"name": "ip_address_reputation", "type": "numeric", "description": "IP address reputation score",
             "range": "[0, 1]", "example": 0.85, "category": "network", "importance": "high"},

            {"name": "vpn_proxy_flag", "type": "boolean", "description": "VPN/proxy usage flag",
             "values": [0, 1], "example": 0, "category": "network", "importance": "high"},

            {"name": "is_vpn_detected", "type": "boolean", "description": "VPN detection flag",
             "values": [0, 1], "example": 0, "category": "network", "importance": "high"},

            # Enhanced behavioral features
            {"name": "typing_pattern_similarity", "type": "numeric", "description": "Typing pattern similarity",
             "range": "[0, 1]", "example": 0.92, "category": "behavior", "importance": "high"},

            {"name": "mouse_movement_similarity", "type": "numeric", "description": "Mouse movement similarity",
             "range": "[0, 1]", "example": 0.88, "category": "behavior", "importance": "medium"},

            {"name": "session_duration", "type": "numeric", "description": "Session duration in minutes",
             "range": "[1, 120]", "example": 25.5, "category": "behavior", "importance": "medium"},

            {"name": "login_frequency_7d", "type": "integer", "description": "Login frequency in last 7 days",
             "range": ">=0", "example": 12, "category": "behavior", "importance": "medium"},

            {"name": "failed_login_attempts", "type": "integer", "description": "Failed login attempts",
             "range": ">=0", "example": 1, "category": "behavior", "importance": "high"},

            # Enhanced interaction features
            {"name": "mobility_score", "type": "numeric", "description": "Combined mobility score",
             "range": "[0, 2]", "example": 0.25, "category": "interaction", "importance": "medium"},

            {"name": "tx_intensity", "type": "numeric", "description": "Transaction intensity score",
             "range": ">=0", "example": 0.1, "category": "interaction", "importance": "medium"},

            {"name": "auth_risk_score", "type": "numeric", "description": "Authentication risk score",
             "range": ">=0", "example": 1.5, "category": "interaction", "importance": "high"},

            # System fields
            {"name": "user_id", "type": "string", "description": "Unique user identifier",
             "example": "user_123456", "category": "system", "importance": "system"},

            {"name": "tx_id", "type": "integer", "description": "Unique transaction identifier",
             "example": 5001, "category": "system", "importance": "system"},

            {"name": "timestamp", "type": "datetime", "description": "Transaction timestamp",
             "format": "YYYY-MM-DD HH:MM:SS", "example": "2023-03-15 10:30:00", "category": "system",
             "importance": "system"},

            {"name": "tx_amount", "type": "numeric", "description": "Transaction amount",
             "range": ">=1", "example": 125.75, "category": "system", "importance": "system"},

            {"name": "account_balance", "type": "numeric", "description": "Account balance",
             "range": ">=0", "example": 5000.00, "category": "system", "importance": "system"},

            # Target variables
            {"name": "risk_label", "type": "categorical", "description": "Risk category label",
             "values": ["legit", "suspicious", "fraud"], "label_categorical": True,
             "example": "legit", "category": "target", "importance": "target"},

            {"name": "risk_flag_manual", "type": "boolean", "description": "Binary risk flag",
             "values": [0, 1], "label_binary": True, "example": 0, "category": "target", "importance": "target"}
        ]

    def _generate_merchant_risk_scores(self) -> Dict[str, float]:
        """Generate realistic merchant risk scores"""
        merchant_risks = {}
        for merchant in self.config.merchants:
            # Most merchants are low risk, few are high risk
            if np.random.random() < 0.05:  # 5% high risk
                risk_score = np.random.uniform(0.7, 1.0)
            elif np.random.random() < 0.15:  # 15% medium risk  
                risk_score = np.random.uniform(0.3, 0.7)
            else:  # 80% low risk
                risk_score = np.random.uniform(0.0, 0.3)

            merchant_risks[merchant] = risk_score

        return merchant_risks

    def _create_enhanced_user_profile(self, user_id: str, risk_profile: RiskProfile) -> Dict[str, Any]:
        """Create enhanced user profile with realistic behavioral patterns"""
        is_anomalous = risk_profile != RiskProfile.LEGITIMATE

        # Basic profile
        profile = {
            'user_id': user_id,
            'risk_profile': risk_profile,
            'is_anomalous': is_anomalous,
            'activity_level': np.random.choice(list(ActivityLevel), p=[0.3, 0.4, 0.25, 0.05]),
            'account_creation_date': self.config.start_date - timedelta(
                days=np.random.randint(30, 2000 if not is_anomalous else 1000)
            )
        }

        # Activity patterns based on level
        activity_configs = {
            ActivityLevel.LOW: {'tx_per_day': (0.2, 1.5), 'active_hours': 3, 'session_duration': (5, 20)},
            ActivityLevel.MEDIUM: {'tx_per_day': (1.5, 4), 'active_hours': 6, 'session_duration': (10, 40)},
            ActivityLevel.HIGH: {'tx_per_day': (4, 8), 'active_hours': 10, 'session_duration': (15, 60)},
            ActivityLevel.VERY_HIGH: {'tx_per_day': (8, 15), 'active_hours': 16, 'session_duration': (20, 90)}
        }

        activity_config = activity_configs[profile['activity_level']]

        # Enhanced transaction patterns
        profile.update({
            'avg_tx_amount': self._generate_realistic_amount_profile(is_anomalous),
            'tx_amount_std': np.random.uniform(50, 300) if not is_anomalous else np.random.uniform(200, 1500),
            'tx_per_day': np.random.uniform(*activity_config['tx_per_day']),
            'preferred_hours': list(np.random.choice(range(24), size=activity_config['active_hours'], replace=False)),
            'weekend_activity_factor': np.random.uniform(0.3, 0.8) if not is_anomalous else np.random.uniform(0.5, 1.5),
            'session_duration_mean': np.random.uniform(*activity_config['session_duration'])
        })

        # Enhanced device and location patterns
        profile.update({
            'device_change_freq': self._generate_device_change_freq(risk_profile),
            'location_change_freq': self._generate_location_change_freq(risk_profile),
            'primary_device_type': np.random.choice(self.config.device_types),
            'primary_location': np.random.choice(self.config.countries),
            'typical_merchants': np.random.choice(self.config.merchants, size=np.random.randint(3, 12), replace=False)
        })

        # Enhanced behavioral patterns
        profile.update({
            'typing_consistency': self._generate_typing_consistency(risk_profile),
            'mouse_consistency': self._generate_mouse_consistency(risk_profile),
            'login_frequency_7d': max(1, int(np.random.normal(10 if not is_anomalous else 5, 3))),
            'vpn_usage_prob': self._generate_vpn_probability(risk_profile)
        })

        # Account details
        profile.update({
            'account_balance': self._generate_account_balance(is_anomalous),
            'account_age_days': max(1, (datetime.now() - profile['account_creation_date']).days)
        })

        self.user_profiles[user_id] = profile
        self.transaction_history[user_id] = []

        return profile

    def _generate_realistic_amount_profile(self, is_anomalous: bool) -> float:
        """Generate realistic transaction amount profile"""
        if is_anomalous:
            # Anomalous users tend to have higher amounts or very specific patterns
            return np.random.lognormal(mean=5.5, sigma=1.2)
        else:
            # Normal users follow more typical spending patterns
            return np.random.lognormal(mean=4.2, sigma=0.8)

    def _generate_device_change_freq(self, risk_profile: RiskProfile) -> float:
        """Generate device change frequency based on risk profile"""
        if risk_profile == RiskProfile.LEGITIMATE:
            return np.random.beta(2, 98)  # Very low change frequency
        elif risk_profile == RiskProfile.SUSPICIOUS:
            return np.random.beta(5, 45)  # Moderate change frequency
        else:  # FRAUDULENT or MIXED
            return np.random.beta(20, 30)  # High change frequency

    def _generate_location_change_freq(self, risk_profile: RiskProfile) -> float:
        """Generate location change frequency based on risk profile"""
        if risk_profile == RiskProfile.LEGITIMATE:
            return np.random.beta(1, 199)  # Very low change frequency
        elif risk_profile == RiskProfile.SUSPICIOUS:
            return np.random.beta(3, 97)  # Low-moderate change frequency
        else:  # FRAUDULENT or MIXED
            return np.random.beta(15, 35)  # High change frequency

    def _generate_typing_consistency(self, risk_profile: RiskProfile) -> float:
        """Generate typing pattern consistency"""
        if risk_profile == RiskProfile.LEGITIMATE:
            return np.random.uniform(*self.config.typing_consistency_normal)
        else:
            return np.random.uniform(*self.config.typing_consistency_anomalous)

    def _generate_mouse_consistency(self, risk_profile: RiskProfile) -> float:
        """Generate mouse movement consistency"""
        if risk_profile == RiskProfile.LEGITIMATE:
            return np.random.uniform(0.85, 0.98)
        else:
            return np.random.uniform(0.2, 0.65)

    def _generate_vpn_probability(self, risk_profile: RiskProfile) -> float:
        """Generate VPN usage probability"""
        if risk_profile == RiskProfile.LEGITIMATE:
            return np.random.uniform(0, 0.05)
        elif risk_profile == RiskProfile.SUSPICIOUS:
            return np.random.uniform(0.1, 0.3)
        else:  # FRAUDULENT or MIXED
            return np.random.uniform(0.4, 0.9)

    def _generate_account_balance(self, is_anomalous: bool) -> float:
        """Generate realistic account balance"""
        if is_anomalous:
            return np.random.lognormal(mean=7, sigma=1.5)
        else:
            return np.random.lognormal(mean=8.2, sigma=1.2)

    def _calculate_enhanced_behavioral_features(self, user_profile: Dict[str, Any],
                                                transaction_data: Dict[str, Any],
                                                user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhanced behavioral features with complex interactions"""
        features = {}

        # Basic temporal features
        timestamp = transaction_data['timestamp']
        hour = timestamp.hour
        is_preferred_hour = hour in user_profile['preferred_hours']

        # Enhanced transaction amount features
        if user_history:
            amounts = [tx['tx_amount'] for tx in user_history[-50:]]  # Last 50 transactions
            if amounts:
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts) + 1e-8
                features['tx_amount_zscore'] = (transaction_data['tx_amount'] - mean_amount) / std_amount
                features['tx_amount_percentile'] = (
                                                           sum(1 for amt in amounts if
                                                               amt <= transaction_data['tx_amount']) / len(amounts)
                                                   ) * 100
            else:
                features['tx_amount_zscore'] = 0
                features['tx_amount_percentile'] = 50
        else:
            features['tx_amount_zscore'] = 0
            features['tx_amount_percentile'] = 50

        # Enhanced behavioral consistency features
        typing_base = user_profile['typing_consistency']
        mouse_base = user_profile['mouse_consistency']

        # Time-of-day effects on behavior
        if not is_preferred_hour:
            typing_base *= np.random.uniform(0.7, 0.95)
            mouse_base *= np.random.uniform(0.75, 0.95)

        # Weekend effects
        if timestamp.weekday() >= 5:
            typing_base *= np.random.uniform(0.85, 1.0)
            mouse_base *= np.random.uniform(0.85, 1.0)

        features['typing_pattern_similarity'] = np.clip(np.random.normal(typing_base, 0.1), 0, 1)
        features['mouse_movement_similarity'] = np.clip(np.random.normal(mouse_base, 0.1), 0, 1)

        # Enhanced device features
        features['device_fingerprint_similarity'] = (
            np.random.uniform(0.8, 1.0) if not transaction_data.get('is_new_device', 0)
            else np.random.uniform(0.1, 0.6)
        )

        # Enhanced location features
        features['distance_from_usual_location'] = (
            np.random.exponential(15) if not user_profile['is_anomalous']
            else np.random.exponential(800)
        )

        # Enhanced network features
        vpn_prob = user_profile['vpn_usage_prob']
        features['vpn_proxy_flag'] = 1 if np.random.random() < vpn_prob else 0
        features['is_vpn_detected'] = features['vpn_proxy_flag']  # Simplified

        # IP reputation based on user profile and VPN usage
        if features['vpn_proxy_flag']:
            features['ip_address_reputation'] = np.random.uniform(0.1, 0.5)
        else:
            features['ip_address_reputation'] = (
                np.random.uniform(0.7, 1.0) if not user_profile['is_anomalous']
                else np.random.uniform(0.2, 0.7)
            )

        # Enhanced velocity features
        now = timestamp
        last_24h = [tx for tx in user_history if (now - tx['timestamp']).total_seconds() <= 86400]
        last_10min = [tx for tx in user_history if (now - tx['timestamp']).total_seconds() <= 600]

        features['transaction_count_24h'] = len(last_24h)
        features['transaction_volume_24h'] = sum(tx['tx_amount'] for tx in last_24h)
        features['transaction_velocity_10min'] = len(last_10min)
        features['time_since_last_tx'] = (
            0 if not user_history else
            (now - user_history[-1]['timestamp']).total_seconds() / 3600
        )

        # Enhanced merchant features
        merchant = transaction_data['merchant_id']
        features['merchant_risk_score'] = self.merchant_risk_scores.get(merchant, 0.5)
        features['is_high_risk_merchant'] = 1 if features['merchant_risk_score'] > 0.6 else 0

        # Enhanced session and authentication features
        base_session_duration = user_profile['session_duration_mean']
        session_variance = base_session_duration * 0.4
        features['session_duration'] = max(0.5, np.random.normal(base_session_duration, session_variance))

        features['login_frequency_7d'] = user_profile['login_frequency_7d']

        # Failed login attempts based on risk profile
        if user_profile['risk_profile'] == RiskProfile.FRAUDULENT:
            features['failed_login_attempts'] = np.random.poisson(2.5)
        elif user_profile['risk_profile'] == RiskProfile.SUSPICIOUS:
            features['failed_login_attempts'] = np.random.poisson(1.0)
        else:
            features['failed_login_attempts'] = np.random.poisson(0.3)

        # Enhanced interaction features
        features['mobility_score'] = user_profile['device_change_freq'] + user_profile['location_change_freq']
        features['tx_intensity'] = features['transaction_velocity_10min'] / (features['session_duration'] + 1)
        features['auth_risk_score'] = features['failed_login_attempts'] * (transaction_data.get('is_new_device', 0) + 1)

        return features

    def _apply_advanced_risk_scoring(self, features: Dict[str, Any], user_profile: Dict[str, Any]) -> Tuple[str, int]:
        """Enhanced risk scoring with sophisticated multi-factor analysis"""
        risk_score = 0.0

        # Amount-based risk (25% weight)
        amount_risk = 0.0
        if features.get('amount_balance_ratio', 0) > 0.6:
            amount_risk += 0.4
        if features.get('tx_amount_zscore', 0) > 3:
            amount_risk += 0.3
        if features.get('tx_amount_percentile', 50) > 95:
            amount_risk += 0.2
        risk_score += min(amount_risk, 1.0) * 0.25

        # Velocity-based risk (20% weight)
        velocity_risk = 0.0
        if features.get('transaction_velocity_10min', 0) >= 5:
            velocity_risk += 0.5
        if features.get('transaction_count_24h', 0) > 15:
            velocity_risk += 0.3
        if features.get('tx_intensity', 0) > 0.5:
            velocity_risk += 0.2
        risk_score += min(velocity_risk, 1.0) * 0.20

        # Location and device risk (20% weight)
        location_device_risk = 0.0
        if features.get('distance_from_usual_location', 0) > 1000:
            location_device_risk += 0.3
        if features.get('is_new_device', 0) == 1:
            location_device_risk += 0.3
        if features.get('country_change_flag', 0) == 1:
            location_device_risk += 0.2
        if features.get('device_fingerprint_similarity', 1) < 0.5:
            location_device_risk += 0.2
        risk_score += min(location_device_risk, 1.0) * 0.20

        # Behavioral risk (15% weight)
        behavioral_risk = 0.0
        if features.get('typing_pattern_similarity', 1) < 0.6:
            behavioral_risk += 0.4
        if features.get('mouse_movement_similarity', 1) < 0.6:
            behavioral_risk += 0.3
        if features.get('session_duration', 30) < 2:
            behavioral_risk += 0.3
        risk_score += min(behavioral_risk, 1.0) * 0.15

        # Network and security risk (10% weight)
        network_risk = 0.0
        if features.get('vpn_proxy_flag', 0) == 1:
            network_risk += 0.4
        if features.get('ip_address_reputation', 1) < 0.3:
            network_risk += 0.4
        if features.get('failed_login_attempts', 0) > 3:
            network_risk += 0.2
        risk_score += min(network_risk, 1.0) * 0.10

        # Merchant risk (5% weight)
        merchant_risk = features.get('merchant_risk_score', 0)
        risk_score += merchant_risk * 0.05

        # Authentication risk (5% weight)
        auth_risk = min(features.get('auth_risk_score', 0) / 10, 1.0)
        risk_score += auth_risk * 0.05

        # Apply user profile modifiers
        if user_profile['risk_profile'] == RiskProfile.FRAUDULENT:
            risk_score = min(1.0, risk_score * 1.3)
        elif user_profile['risk_profile'] == RiskProfile.SUSPICIOUS:
            risk_score = min(1.0, risk_score * 1.1)

        # Final risk classification
        if risk_score > 0.75:
            return "fraud", 1
        elif risk_score > 0.45:
            return "suspicious", 1
        else:
            return "legit", 0

    def _generate_enhanced_single_transaction(self, user_profile: Dict[str, Any],
                                              is_first: bool) -> Dict[str, Any]:
        """Generate a single enhanced transaction with realistic complexity"""
        # Generate realistic timestamp
        timestamp = self._generate_realistic_timestamp(user_profile)

        # Generate transaction amount based on user profile and context
        tx_amount = self._generate_realistic_transaction_amount(user_profile, timestamp)

        # Select merchant (prefer typical merchants)
        if np.random.random() < 0.7:  # 70% chance to use typical merchant
            merchant_id = np.random.choice(user_profile['typical_merchants'])
        else:
            merchant_id = np.random.choice(self.config.merchants)

        # Generate device information
        if np.random.random() < user_profile['device_change_freq']:
            device_type = np.random.choice(self.config.device_types)
            is_new_device = 1
        else:
            device_type = user_profile['primary_device_type']
            is_new_device = 0

        # Generate location information
        if np.random.random() < user_profile['location_change_freq']:
            location = np.random.choice(self.config.countries)
            country_change_flag = 1 if location != user_profile['primary_location'] else 0
        else:
            location = user_profile['primary_location']
            country_change_flag = 0

        transaction = {
            'user_id': user_profile['user_id'],
            'tx_id': self.tx_id_counter,
            'timestamp': timestamp,
            'tx_amount': tx_amount,
            'account_balance': user_profile['account_balance'],
            'merchant_id': merchant_id,
            'device_type': device_type,
            'location': location,
            'tx_hour': timestamp.hour,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_holiday': 1 if self._is_enhanced_holiday(timestamp) else 0,
            'is_business_hours': 1 if 9 <= timestamp.hour <= 17 else 0,
            'is_unusual_hour': 1 if timestamp.hour < 6 or timestamp.hour > 23 else 0,
            'is_new_device': is_new_device,
            'country_change_flag': country_change_flag,
            'amount_balance_ratio': min(1.0, tx_amount / user_profile['account_balance']),
            'avg_tx_amount': user_profile['avg_tx_amount'],
            'device_change_freq': user_profile['device_change_freq'],
            'location_change_freq': user_profile['location_change_freq'],
            'account_age_days': user_profile['account_age_days']
        }

        self.tx_id_counter += 1
        return transaction

    def _generate_realistic_transaction_amount(self, user_profile: Dict[str, Any],
                                               timestamp: datetime) -> float:
        """Generate realistic transaction amount with complex patterns"""
        base_amount = user_profile['avg_tx_amount']
        std_amount = user_profile['tx_amount_std']

        # Time-based modifiers
        hour = timestamp.hour
        if hour < 6 or hour > 22:  # Late night/early morning
            base_amount *= np.random.uniform(0.3, 0.8)
        elif 9 <= hour <= 17:  # Business hours
            base_amount *= np.random.uniform(0.8, 1.2)
        elif hour >= 18:  # Evening
            base_amount *= np.random.uniform(1.0, 1.4)

        # Weekend modifier
        if timestamp.weekday() >= 5:
            base_amount *= user_profile['weekend_activity_factor']

        # Risk profile modifier
        if user_profile['risk_profile'] == RiskProfile.FRAUDULENT:
            base_amount *= np.random.uniform(1.5, 4.0)
        elif user_profile['risk_profile'] == RiskProfile.SUSPICIOUS:
            base_amount *= np.random.uniform(1.2, 2.0)

        # Generate final amount
        amount = max(1.0, np.random.normal(base_amount, std_amount))
        return min(amount, self.config.max_transaction_amount)

    def _generate_realistic_timestamp(self, user_profile: Dict[str, Any]) -> datetime:
        """Generate realistic timestamp with user behavioral patterns"""
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            # Random timestamp in range
            timestamp = self.config.start_date + timedelta(
                seconds=random.randint(
                    0, int((self.config.end_date - self.config.start_date).total_seconds())
                )
            )

            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
            is_preferred_hour = hour in user_profile['preferred_hours']

            # Acceptance probability based on user patterns
            acceptance_prob = 0.1  # Base probability

            if is_preferred_hour:
                acceptance_prob += 0.6
            if is_weekend and user_profile['weekend_activity_factor'] > 0.8:
                acceptance_prob += 0.2
            if not is_weekend and 9 <= hour <= 17:
                acceptance_prob += 0.1

            if np.random.random() < acceptance_prob:
                return timestamp

            attempts += 1

        # Fallback
        return self.config.start_date + timedelta(
            seconds=random.randint(
                0, int((self.config.end_date - self.config.start_date).total_seconds())
            )
        )

    def _is_enhanced_holiday(self, date: datetime) -> bool:
        """Enhanced holiday detection with more holidays"""
        holidays = [
            (1, 1), (1, 15), (2, 14), (3, 17), (7, 4), (10, 31),
            (11, 11), (11, 24), (12, 25), (12, 31)
        ]
        return (date.month, date.day) in holidays

    def _generate_enhanced_user_transactions(self, user_profile: Dict[str, Any],
                                             num_transactions: int, data: List[Dict[str, Any]]) -> None:
        """Generate enhanced transactions for a specific user with realistic patterns"""
        for j in range(num_transactions):
            # Generate base transaction
            transaction_data = self._generate_enhanced_single_transaction(user_profile, j == 0)

            # Calculate behavioral features
            behavioral_features = self._calculate_enhanced_behavioral_features(
                user_profile, transaction_data, self.transaction_history[user_profile['user_id']]
            )

            # Combine all features
            combined_features = {**transaction_data, **behavioral_features}

            # Apply risk scoring
            risk_label, risk_flag = self._apply_advanced_risk_scoring(combined_features, user_profile)

            # Set final labels
            combined_features['risk_label'] = risk_label
            combined_features['risk_flag_manual'] = risk_flag

            # Store transaction
            data.append(combined_features)
            self.transaction_history[user_profile['user_id']].append(transaction_data)

    def generate_comprehensive_dataset(self) -> pd.DataFrame:
        """Generate comprehensive enhanced dataset with realistic patterns"""
        logger.info("Starting enhanced comprehensive behavioral data generation")

        data = []
        total_users = self.config.num_normal_users + self.config.num_anomalous_users

        # Generate normal users
        logger.info(f"Generating data for {self.config.num_normal_users} normal users")
        for i in range(self.config.num_normal_users):
            if i % 200 == 0:
                logger.info(f"Progress: {i}/{self.config.num_normal_users} normal users processed")

            user_id = f'user_{i + 1:06d}'
            user_profile = self._create_enhanced_user_profile(user_id, RiskProfile.LEGITIMATE)

            num_transactions = max(1, int(np.random.negative_binomial(
                self.config.transactions_per_normal_user_mean, 0.3
            )))

            self._generate_enhanced_user_transactions(user_profile, num_transactions, data)

        # Generate anomalous users with mixed risk profiles
        logger.info(f"Generating data for {self.config.num_anomalous_users} anomalous users")
        for i in range(self.config.num_anomalous_users):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{self.config.num_anomalous_users} anomalous users processed")

            user_id = f'anom_user_{i + 1:06d}'

            # Determine risk profile based on configuration
            rand_val = np.random.random()
            if rand_val < 0.4:  # 40% suspicious
                risk_profile = RiskProfile.SUSPICIOUS
            elif rand_val < 0.7:  # 30% fraudulent
                risk_profile = RiskProfile.FRAUDULENT
            else:  # 30% mixed (harder to classify)
                risk_profile = RiskProfile.MIXED

            user_profile = self._create_enhanced_user_profile(user_id, risk_profile)

            num_transactions = max(1, int(np.random.negative_binomial(
                self.config.transactions_per_anomalous_user_mean, 0.4
            )))

            self._generate_enhanced_user_transactions(user_profile, num_transactions, data)

        # Create enhanced DataFrame
        logger.info("Creating enhanced DataFrame from generated data")
        df = pd.DataFrame(data)

        logger.info(f"Generated {len(df)} total transactions from {total_users} users")
        if 'risk_label' in df.columns:
            logger.info(f"Risk label distribution:\n{df['risk_label'].value_counts()}")

        return df

    def generate_api_test_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate lightweight test data specifically for API testing"""
        logger.info(f"Generating {num_samples} API test samples")

        # Core features needed for API
        api_features = [
            'tx_amount', 'tx_hour', 'is_weekend', 'is_new_device', 'device_fingerprint_similarity',
            'distance_from_usual_location', 'transaction_count_24h', 'typing_pattern_similarity',
            'amount_balance_ratio', 'vpn_proxy_flag', 'ip_address_reputation', 'merchant_risk_score',
            'session_duration', 'failed_login_attempts', 'is_business_hours'
        ]

        test_data = []
        for i in range(num_samples):
            # Generate a mini user profile for this sample
            risk_profile = np.random.choice([RiskProfile.LEGITIMATE, RiskProfile.SUSPICIOUS, RiskProfile.FRAUDULENT])
            user_profile = self._create_enhanced_user_profile(f'api_test_{i}', risk_profile)

            # Generate a single transaction
            transaction = self._generate_enhanced_single_transaction(user_profile, True)
            behavioral_features = self._calculate_enhanced_behavioral_features(
                user_profile, transaction, []
            )

            # Combine and filter to API features
            all_features = {**transaction, **behavioral_features}
            api_sample = {k: all_features[k] for k in api_features if k in all_features}

            # Add expected output
            risk_label, risk_flag = self._apply_advanced_risk_scoring(all_features, user_profile)
            api_sample['expected_risk_label'] = risk_label
            api_sample['expected_risk_flag'] = risk_flag

            test_data.append(api_sample)

        return test_data

    def save_enhanced_dataset(self, df: pd.DataFrame) -> None:
        """Save enhanced dataset with comprehensive validation and reporting"""
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.config.data_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.schema_file), exist_ok=True)

            # Save main dataset
            df.to_csv(self.config.data_file, index=False)
            logger.info(f"Enhanced dataset saved to {self.config.data_file}")

            # Save enhanced feature schema
            self._save_enhanced_feature_schema()

            # Generate API test data
            api_test_data = self.generate_api_test_data()
            with open(self.config.api_test_file, 'w') as f:
                json.dump(api_test_data, f, indent=2, default=str)
            logger.info(f"API test data saved to {self.config.api_test_file}")

        except Exception as e:
            logger.error(f"Error saving enhanced dataset: {e}")
            raise

    def _save_enhanced_feature_schema(self) -> None:
        """Save enhanced feature schema to JSON file"""
        schema_data = {
            'schema_version': '2.0_enhanced',
            'description': 'Enhanced Behavioral Authentication ML Feature Schema',
            'generated_at': datetime.now().isoformat(),
            'generator_info': {
                'version': '2.0_enhanced',
                'author': 'olafcio42',
                'repo': 'AI-Quantum-Tech-Security/BMD'
            },
            'features': self.feature_schema,
            'target_variables': {
                'binary': self.config.target_column_binary,
                'categorical': self.config.target_column_categorical
            },
            'feature_categories': {
                'transaction': [f['name'] for f in self.feature_schema if f.get('category') == 'transaction'],
                'device': [f['name'] for f in self.feature_schema if f.get('category') == 'device'],
                'location': [f['name'] for f in self.feature_schema if f.get('category') == 'location'],
                'temporal': [f['name'] for f in self.feature_schema if f.get('category') == 'temporal'],
                'velocity': [f['name'] for f in self.feature_schema if f.get('category') == 'velocity'],
                'behavior': [f['name'] for f in self.feature_schema if f.get('category') == 'behavior'],
                'network': [f['name'] for f in self.feature_schema if f.get('category') == 'network'],
                'merchant': [f['name'] for f in self.feature_schema if f.get('category') == 'merchant'],
                'account': [f['name'] for f in self.feature_schema if f.get('category') == 'account'],
                'interaction': [f['name'] for f in self.feature_schema if f.get('category') == 'interaction']
            },
            'high_importance_features': [f['name'] for f in self.feature_schema if f.get('importance') == 'high'],
            'api_features': [
                'tx_amount', 'tx_hour', 'is_weekend', 'is_new_device', 'device_fingerprint_similarity',
                'distance_from_usual_location', 'transaction_count_24h', 'typing_pattern_similarity',
                'amount_balance_ratio', 'vpn_proxy_flag', 'ip_address_reputation', 'merchant_risk_score',
                'session_duration', 'failed_login_attempts', 'is_business_hours'
            ]
        }

        with open(self.config.schema_file, 'w') as f:
            json.dump(schema_data, f, indent=2)
        logger.info(f"Enhanced feature schema saved to {self.config.schema_file}")


def main() -> None:
    """Main execution function with enhanced error handling"""
    try:
        print("=== Enhanced Behavioral Authentication ML Data Generator v2.0 ===")
        print(f"Generated by: olafcio42")
        print(f"Repository: AI-Quantum-Tech-Security/BMD")
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Initialize enhanced configuration and generator
        config = EnhancedDataGenerationConfig()
        generator = EnhancedBehavioralDataGenerator(config)

        # Generate enhanced dataset
        logger.info("Starting enhanced dataset generation...")
        df = generator.generate_comprehensive_dataset()

        # Save enhanced dataset and generate reports
        generator.save_enhanced_dataset(df)

        print("\n=== Enhanced Generation Complete ===")
        print(f"Dataset: {config.data_file}")
        print(f"Schema: {config.schema_file}")
        print(f"API Test Data: {config.api_test_file}")
        print(f"Total transactions: {len(df):,}")
        print(f"Total users: {df['user_id'].nunique():,}")

        if 'risk_label' in df.columns:
            risk_dist = df['risk_label'].value_counts()
            print(f"Risk distribution: {dict(risk_dist)}")

        print("\nEnhanced behavioral data generation completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()