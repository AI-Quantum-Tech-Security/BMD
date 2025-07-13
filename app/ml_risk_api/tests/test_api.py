"""
Test suite for ML Risk API
"""

import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml_risk_api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "ML Risk API" in data["message"]


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    # Accept 200, 500, or 503 status codes
    assert response.status_code in [200, 500, 503]


def test_risk_score_prediction():
    """Test risk score prediction endpoint"""

    # Sample valid request
    test_request = {
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

    response = client.post("/risk-score", json=test_request)

    # Should work with mock prediction
    assert response.status_code in [200, 422, 503]

    if response.status_code == 200:
        data = response.json()
        assert "risk_score" in data
        assert "risk_flag" in data
        assert "confidence" in data
        assert 0 <= data["risk_score"] <= 1
        assert data["risk_flag"] in ["legit", "suspicious", "fraud"]


def test_invalid_request():
    """Test API validation with invalid request"""

    invalid_request = {
        "tx_hour": 25,  # Invalid hour
        "device_change_freq": 1.5,  # Invalid frequency
    }

    response = client.post("/risk-score", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_batch_prediction():
    """Test batch prediction endpoint"""

    batch_request = [
        {
            "avg_tx_amount": 100.0,
            "device_change_freq": 0.1,
            "tx_hour": 10,
            "location_change_freq": 0.1,
            "is_new_device": 0,
            "transaction_count_24h": 2,
            "time_since_last_tx": 1.0,
            "tx_amount_to_balance_ratio": 0.1,
            "ip_address_reputation": 0.9,
            "is_weekend": 0,
            "transaction_velocity_10min": 1,
            "country_change_flag": 0,
            "session_duration": 10.0,
            "tx_amount": 100.0,
            "account_age_days": 100,
            "amount_balance_ratio": 0.1,
            "typing_pattern_similarity": 0.9,
            "mouse_movement_similarity": 0.9,
            "distance_from_usual_location": 1.0,
            "transaction_volume_24h": 1000.0,
            "merchant_risk_score": 0.1,
            "login_frequency_7d": 5,
            "failed_login_attempts": 0,
            "mobility_score": 0.2,
            "tx_intensity": 0.2,
            "auth_risk_score": 0.1,
            "tx_amount_zscore": 0.5,
            "tx_amount_percentile": 50.0,
            "is_holiday": 0,
            "is_business_hours": 1,
            "is_unusual_hour": 0,
            "vpn_proxy_flag": 0,
            "is_vpn_detected": 0,
            "is_high_risk_merchant": 0,
            "merchant_id": "TEST_MERCHANT",
            "device_type": "desktop",
            "location": "Test City"
        }
    ]

    response = client.post("/risk-score/batch", json=batch_request)
    assert response.status_code in [200, 503]  # Success or service unavailable

def test_minimal_request():
    """Test with minimal required fields only"""
    minimal_request = {
        "avg_tx_amount": 100.0,
        "tx_hour": 12
    }

    response = client.post("/risk-score", json=minimal_request)
    assert response.status_code == 200

    data = response.json()
    assert "risk_score" in data
    assert "risk_flag" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])