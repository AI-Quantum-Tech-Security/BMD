from fastapi.testclient import TestClient
from risk_api import app
import pandas as pd

client = TestClient(app)

# Load synthetic test data
test_data = pd.read_csv('files/synthetic_behavioral_dataset.csv')
test_sample = test_data.iloc[0]


def test_risk_score_endpoint():
    response = client.post(
        "/risk-score",
        json={
            "avg_tx_amount": float(test_sample['avg_tx_amount']),
            "device_change_freq": float(test_sample['device_change_freq']),
            "tx_hour": int(test_sample['tx_hour']),
            "location_change_freq": float(test_sample['location_change_freq']),
            "is_new_device": int(test_sample['is_new_device']),
            "transaction_count_24h": int(test_sample['transaction_count_24h']),
            "time_since_last_tx": float(test_sample['time_since_last_tx']),
            "tx_amount_to_balance_ratio": float(test_sample['tx_amount_to_balance_ratio']),
            "ip_address_reputation": float(test_sample['ip_address_reputation']),
            "is_weekend": int(test_sample['is_weekend']),
            "transaction_velocity_10min": int(test_sample['transaction_velocity_10min']),
            "country_change_flag": int(test_sample['country_change_flag'])
        }
    )

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