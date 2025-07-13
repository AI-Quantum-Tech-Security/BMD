#!/bin/bash

# Enhanced ML Risk API Demo Script
# Demonstrates the functionality of the consolidated risk_api.py

echo "🚀 Enhanced ML Risk API Demo"
echo "==============================================="

# Configuration
API_URL="https://localhost:8443"  # Default HTTPS URL
API_URL_HTTP="http://localhost:8443"  # Fallback HTTP URL

# Check if API is running
echo "1. Checking API availability..."

# Try HTTPS first, fallback to HTTP
if curl -k -s "$API_URL/health" > /dev/null 2>&1; then
    BASE_URL="$API_URL"
    echo "   ✅ API available on HTTPS: $BASE_URL"
elif curl -s "$API_URL_HTTP/health" > /dev/null 2>&1; then
    BASE_URL="$API_URL_HTTP"
    echo "   ✅ API available on HTTP: $BASE_URL"
else
    echo "   ❌ API not available. Start the API first:"
    echo "      python files/risk_api.py"
    exit 1
fi

echo
echo "2. API Information..."
curl -k -s "$BASE_URL/" | jq '.' 2>/dev/null || curl -k -s "$BASE_URL/"

echo
echo "3. Health Check..."
curl -k -s "$BASE_URL/health" | jq '.' 2>/dev/null || curl -k -s "$BASE_URL/health"

echo
echo "4. Model Information..."
curl -k -s "$BASE_URL/model/info" | jq '.' 2>/dev/null || curl -k -s "$BASE_URL/model/info"

echo
echo "5. Low Risk Transaction Prediction..."
curl -k -s -X POST "$BASE_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "avg_tx_amount": 100.0,
    "device_change_freq": 0.0,
    "tx_hour": 14,
    "location_change_freq": 0.0,
    "is_new_device": 0,
    "transaction_count_24h": 2,
    "time_since_last_tx": 4.0,
    "tx_amount_to_balance_ratio": 0.1,
    "ip_address_reputation": 0.95,
    "is_weekend": 0,
    "transaction_velocity_10min": 1,
    "country_change_flag": 0,
    "session_duration": 10.0,
    "tx_amount": 100.0,
    "merchant_id": "TRUSTED_MERCHANT",
    "device_type": "desktop",
    "location": "New York"
  }' | jq '.' 2>/dev/null || curl -k -s -X POST "$BASE_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "avg_tx_amount": 100.0,
    "device_change_freq": 0.0,
    "tx_hour": 14,
    "location_change_freq": 0.0,
    "is_new_device": 0,
    "transaction_count_24h": 2,
    "time_since_last_tx": 4.0,
    "tx_amount_to_balance_ratio": 0.1,
    "ip_address_reputation": 0.95,
    "is_weekend": 0,
    "transaction_velocity_10min": 1,
    "country_change_flag": 0
  }'

echo
echo "6. High Risk Transaction Prediction..."
curl -k -s -X POST "$BASE_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "avg_tx_amount": 100.0,
    "device_change_freq": 0.8,
    "tx_hour": 3,
    "location_change_freq": 0.7,
    "is_new_device": 1,
    "transaction_count_24h": 15,
    "time_since_last_tx": 0.1,
    "tx_amount_to_balance_ratio": 0.9,
    "ip_address_reputation": 0.2,
    "is_weekend": 1,
    "transaction_velocity_10min": 8,
    "country_change_flag": 1,
    "session_duration": 2.0,
    "tx_amount": 5000.0,
    "merchant_id": "SUSPICIOUS_MERCHANT",
    "device_type": "mobile",
    "location": "Unknown"
  }' | jq '.' 2>/dev/null || curl -k -s -X POST "$BASE_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "avg_tx_amount": 100.0,
    "device_change_freq": 0.8,
    "tx_hour": 3,
    "location_change_freq": 0.7,
    "is_new_device": 1,
    "transaction_count_24h": 15,
    "time_since_last_tx": 0.1,
    "tx_amount_to_balance_ratio": 0.9,
    "ip_address_reputation": 0.2,
    "is_weekend": 1,
    "transaction_velocity_10min": 8,
    "country_change_flag": 1
  }'

echo
echo "7. Batch Prediction..."
curl -k -s -X POST "$BASE_URL/risk-score/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "avg_tx_amount": 200.0,
      "device_change_freq": 0.1,
      "tx_hour": 10,
      "location_change_freq": 0.05,
      "is_new_device": 0,
      "transaction_count_24h": 3,
      "time_since_last_tx": 2.0,
      "tx_amount_to_balance_ratio": 0.15,
      "ip_address_reputation": 0.9,
      "is_weekend": 0,
      "transaction_velocity_10min": 1,
      "country_change_flag": 0
    },
    {
      "avg_tx_amount": 100.0,
      "device_change_freq": 0.5,
      "tx_hour": 23,
      "location_change_freq": 0.4,
      "is_new_device": 1,
      "transaction_count_24h": 10,
      "time_since_last_tx": 0.5,
      "tx_amount_to_balance_ratio": 0.7,
      "ip_address_reputation": 0.3,
      "is_weekend": 1,
      "transaction_velocity_10min": 5,
      "country_change_flag": 1
    }
  ]' | jq '.' 2>/dev/null || curl -k -s -X POST "$BASE_URL/risk-score/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "avg_tx_amount": 200.0,
      "device_change_freq": 0.1,
      "tx_hour": 10,
      "location_change_freq": 0.05,
      "is_new_device": 0,
      "transaction_count_24h": 3,
      "time_since_last_tx": 2.0,
      "tx_amount_to_balance_ratio": 0.15,
      "ip_address_reputation": 0.9,
      "is_weekend": 0,
      "transaction_velocity_10min": 1,
      "country_change_flag": 0
    }
  ]'

echo
echo "8. API Metrics..."
curl -k -s "$BASE_URL/metrics" | jq '.' 2>/dev/null || curl -k -s "$BASE_URL/metrics"

echo
echo "9. Testing Invalid Request (should return 422)..."
curl -k -s -X POST "$BASE_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_hour": 25,
    "invalid_field": "test"
  }' | jq '.' 2>/dev/null || curl -k -s -X POST "$BASE_URL/risk-score" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_hour": 25,
    "invalid_field": "test"
  }'

echo
echo "10. Testing Rate Limiting..."
echo "    Making multiple rapid requests..."
for i in {1..5}; do
  echo -n "    Request $i: "
  response=$(curl -k -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/risk-score" \
    -H "Content-Type: application/json" \
    -d '{
      "avg_tx_amount": 100.0,
      "tx_hour": 12,
      "device_change_freq": 0.1,
      "location_change_freq": 0.1,
      "is_new_device": 0,
      "transaction_count_24h": 1,
      "time_since_last_tx": 1.0,
      "tx_amount_to_balance_ratio": 0.1,
      "ip_address_reputation": 0.8,
      "is_weekend": 0,
      "transaction_velocity_10min": 1,
      "country_change_flag": 0
    }')
  
  if [ "$response" == "200" ]; then
    echo "✅ Success (200)"
  elif [ "$response" == "429" ]; then
    echo "🚫 Rate Limited (429)"
  else
    echo "❓ Other ($response)"
  fi
  
  sleep 0.1
done

echo
echo "==============================================="
echo "✅ Enhanced ML Risk API Demo Completed!"
echo
echo "Available Endpoints:"
echo "- GET  $BASE_URL/                    - API information"
echo "- GET  $BASE_URL/health              - Health check"
echo "- GET  $BASE_URL/model/info          - Model information"
echo "- POST $BASE_URL/risk-score          - Single prediction"
echo "- POST $BASE_URL/risk-score/batch    - Batch predictions"
echo "- GET  $BASE_URL/metrics             - API metrics"
echo "- GET  $BASE_URL/docs                - Swagger documentation"
echo "- GET  $BASE_URL/redoc               - ReDoc documentation"
echo
echo "Configuration Examples:"
echo "- Development:  python files/risk_api.py"
echo "- HTTPS:        RISK_API_SSL_ENABLED=true python files/risk_api.py"
echo "- Custom Port:  RISK_API_PORT=9443 python files/risk_api.py"
echo "- With API Key: RISK_API_API_KEY=secret python files/risk_api.py"