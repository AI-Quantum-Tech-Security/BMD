#!/bin/bash

# ML Risk API Test Script
echo "🚀 Testing ML Risk API..."

# Set API base URL
API_URL="http://localhost:8000"

echo "1. Testing root endpoint..."
curl -X GET "$API_URL/" | jq

echo -e "\n2. Testing health check..."
curl -X GET "$API_URL/health" | jq

echo -e "\n3. Testing model info..."
curl -X GET "$API_URL/model/info" | jq

echo -e "\n4. Testing risk score prediction..."
curl -X POST "$API_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | jq

echo -e "\n5. Testing invalid request (should return 422)..."
curl -X POST "$API_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_hour": 25,
    "invalid_field": "test"
  }' | jq

echo -e "\n✅ API testing completed!"