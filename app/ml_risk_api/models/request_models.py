"""
Request models for ML Risk API using Pydantic for validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class RiskScoreRequest(BaseModel):
    """
    Request model for risk score prediction
    Contains all 76 behavioral features from the trained XGBoost model
    """

    # Core behavioral features (original 12)
    avg_tx_amount: float = Field(..., ge=0, description="Average transaction amount for the user")
    device_change_freq: float = Field(..., ge=0, le=1, description="Frequency of device changes")
    tx_hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    location_change_freq: float = Field(..., ge=0, le=1, description="Frequency of location changes")
    is_new_device: int = Field(..., ge=0, le=1, description="Boolean flag for new device (0/1)")
    transaction_count_24h: int = Field(..., ge=0, description="Number of transactions in last 24h")
    time_since_last_tx: float = Field(..., ge=0, description="Hours since last transaction")
    tx_amount_to_balance_ratio: float = Field(..., ge=0, le=1, description="Transaction to balance ratio")
    ip_address_reputation: float = Field(..., ge=0, le=1, description="IP reputation score (0-1)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag (0/1)")
    transaction_velocity_10min: int = Field(..., ge=0, description="Transactions in last 10 minutes")
    country_change_flag: int = Field(..., ge=0, le=1, description="Country change flag (0/1)")

    # Enhanced behavioral features
    session_duration: float = Field(..., ge=0, description="Session duration in minutes")
    tx_amount: float = Field(..., ge=0, description="Current transaction amount")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    amount_balance_ratio: float = Field(..., ge=0, le=1, description="Amount to balance ratio")
    typing_pattern_similarity: float = Field(..., ge=0, le=1, description="Typing pattern similarity")
    mouse_movement_similarity: float = Field(..., ge=0, le=1, description="Mouse movement similarity")
    distance_from_usual_location: float = Field(..., ge=0, description="Distance from usual location (km)")
    transaction_volume_24h: float = Field(..., ge=0, description="Total transaction volume in 24h")
    merchant_risk_score: float = Field(..., ge=0, le=1, description="Merchant risk score")
    login_frequency_7d: int = Field(..., ge=0, description="Login frequency in 7 days")
    failed_login_attempts: int = Field(..., ge=0, description="Failed login attempts")
    mobility_score: float = Field(..., ge=0, le=1, description="User mobility score")
    tx_intensity: float = Field(..., ge=0, description="Transaction intensity score")
    auth_risk_score: float = Field(..., ge=0, le=1, description="Authentication risk score")
    tx_amount_zscore: float = Field(..., description="Transaction amount Z-score")
    tx_amount_percentile: float = Field(..., ge=0, le=100, description="Transaction amount percentile")

    # Boolean flags
    is_holiday: int = Field(..., ge=0, le=1, description="Holiday flag (0/1)")
    is_business_hours: int = Field(..., ge=0, le=1, description="Business hours flag (0/1)")
    is_unusual_hour: int = Field(..., ge=0, le=1, description="Unusual hour flag (0/1)")
    vpn_proxy_flag: int = Field(..., ge=0, le=1, description="VPN/Proxy flag (0/1)")
    is_vpn_detected: int = Field(..., ge=0, le=1, description="VPN detected flag (0/1)")
    is_high_risk_merchant: int = Field(..., ge=0, le=1, description="High risk merchant flag (0/1)")

    # Categorical features (will be one-hot encoded)
    merchant_id: str = Field(..., description="Merchant identifier")
    device_type: str = Field(..., description="Device type (mobile/desktop/tablet)")
    location: str = Field(..., description="Transaction location")

    # Engineered interaction features (will be calculated)
    tx_amount_ratio: Optional[float] = Field(None, description="Transaction amount ratio")
    tx_amount_deviation: Optional[float] = Field(None, description="Transaction amount deviation")
    session_tx_intensity: Optional[float] = Field(None, description="Session transaction intensity")
    behavior_volatility: Optional[float] = Field(None, description="Behavioral volatility score")
    mobility_risk: Optional[float] = Field(None, description="Mobility risk score")
    network_risk_combined: Optional[float] = Field(None, description="Combined network risk")
    behavioral_consistency: Optional[float] = Field(None, description="Behavioral consistency score")

    @validator('tx_hour')
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('tx_hour must be between 0 and 23')
        return v

    @validator('device_type')
    def validate_device_type(cls, v):
        allowed_devices = ['mobile', 'desktop', 'tablet', 'other']
        if v.lower() not in allowed_devices:
            raise ValueError(f'device_type must be one of: {allowed_devices}')
        return v.lower()

    @validator('merchant_id')
    def validate_merchant_id(cls, v):
        if len(v) > 50:
            raise ValueError('merchant_id too long (max 50 characters)')
        return v

    class Config:
        schema_extra = {
            "example": {
                "avg_tx_amount": 500.25,
                "device_change_freq": 0.05,
                "tx_hour": 14,
                "location_change_freq": 0.15,
                "is_new_device": 0,
                "transaction_count_24h": 5,
                "time_since_last_tx": 2.5,
                "tx_amount_to_balance_ratio": 0.15,
                "ip_address_reputation": 0.85,
                "is_weekend": 1,
                "transaction_velocity_10min": 2,
                "country_change_flag": 0,
                "session_duration": 15.5,
                "tx_amount": 750.00,
                "account_age_days": 365,
                "amount_balance_ratio": 0.12,
                "typing_pattern_similarity": 0.92,
                "mouse_movement_similarity": 0.88,
                "distance_from_usual_location": 2.5,
                "transaction_volume_24h": 2500.00,
                "merchant_risk_score": 0.2,
                "login_frequency_7d": 12,
                "failed_login_attempts": 0,
                "mobility_score": 0.3,
                "tx_intensity": 0.4,
                "auth_risk_score": 0.1,
                "tx_amount_zscore": 1.2,
                "tx_amount_percentile": 75.5,
                "is_holiday": 0,
                "is_business_hours": 1,
                "is_unusual_hour": 0,
                "vpn_proxy_flag": 0,
                "is_vpn_detected": 0,
                "is_high_risk_merchant": 0,
                "merchant_id": "MERCHANT_001",
                "device_type": "mobile",
                "location": "New York"
            }
        }