from fastapi.testclient import TestClient
from risk_api import app
import pandas as pd

client = TestClient(app)

# Load synthetic test data
test_data = pd.read_csv('files/synthetic_behavioral_dataset.csv')
test_sample = test_data.iloc[0]

# Map dataset columns to API expected features
def create_api_features_from_sample(sample):
    """Map dataset columns to API expected feature names"""
    return {
        "avg_tx_amount": float(sample.get('tx_amount', 500.0)),  # Use current tx_amount as proxy
        "device_change_freq": 0.05,  # Default value as not in dataset
        "tx_hour": int(sample['tx_hour']),
        "location_change_freq": float(sample.get('distance_from_usual_location', 0)) / 1000,  # Normalize distance
        "is_new_device": int(sample['is_new_device']),
        "transaction_count_24h": int(sample['transaction_count_24h']),
        "time_since_last_tx": 2.5,  # Default value as not in dataset
        "tx_amount_to_balance_ratio": float(sample['tx_amount_to_balance_ratio']),
        "ip_address_reputation": float(sample['ip_address_reputation']),
        "is_weekend": int(sample['is_weekend']),
        "transaction_velocity_10min": int(sample['transaction_velocity_10min']),
        "country_change_flag": int(sample['country_change_flag'])
    }


def test_risk_score_endpoint():
    api_features = create_api_features_from_sample(test_sample)
    response = client.post("/risk-score", json=api_features)

    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_flag" in data
    assert "timestamp" in data
    assert "request_id" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_flag"] in ["legit", "suspicious", "fraud"]


def test_invalid_input():
    response = client.post(
        "/risk-score",
        json={
            "avg_tx_amount": "invalid",
            "device_change_freq": 0.05
        }
    )
    assert response.status_code == 422


def test_missing_features():
    response = client.post(
        "/risk-score",
        json={}
    )
    assert response.status_code == 422