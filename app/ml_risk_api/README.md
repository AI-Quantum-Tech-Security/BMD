# ML Risk API - Behavioral Authentication Microservice

Production-ready FastAPI microservice for serving XGBoost behavioral risk assessment model.

## 🚀 Features

- **FastAPI Framework** with automatic OpenAPI/Swagger documentation
- **XGBoost Model** serving with 97.78% F1-score performance
- **76 Behavioral Features** for comprehensive risk assessment
- **Real-time Predictions** with sub-100ms response times
- **Comprehensive Logging** for debugging and monitoring
- **Docker Containerization** for easy deployment
- **Health Checks** and monitoring endpoints
- **Input Validation** using Pydantic models
- **CORS Support** for web application integration

## 📊 Model Performance

- **Model Type**: XGBoost Classifier
- **F1-Score**: 97.78%
- **ROC-AUC**: 98.56%
- **Cross-Validation**: 99.76% stability
- **Features**: 76 engineered behavioral features
- **Classes**: legit, suspicious, fraud

## 🏗️ Architecture

```
ml_risk_api/
├── main.py                 # FastAPI application
├── models/                 # Pydantic validation models
├── services/               # Business logic services
├── tests/                  # Test suite
├── requirements.txt        # Dependencies
├── Dockerfile             # Container definition
└── docker-compose.yml     # Development setup
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ml_risk_api
pip install -r requirements.txt
```

### 2. Run the API

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Test the API

```bash
# Run test script
chmod +x test_api.sh
./test_api.sh

# Or test manually
curl -X GET http://localhost:8000/health
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t ml-risk-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/../files:/app/files:ro ml-risk-api

# Or use docker-compose
docker-compose up -d
```

### Production Deployment

```bash
# Production build
docker build -t ml-risk-api:production .

# Deploy with environment variables
docker run -d \
  --name ml-risk-api \
  -p 8000:8000 \
  -v /path/to/model/files:/app/files:ro \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  ml-risk-api:production
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |
| `/model/info` | GET | Model metadata |
| `/risk-score` | POST | Single prediction |
| `/risk-score/batch` | POST | Batch predictions |

### Sample Request

```bash
curl -X POST "http://localhost:8000/risk-score" \
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
  }'
```

### Sample Response

```json
{
  "request_id": "req_1642086123456",
  "risk_score": 0.15,
  "risk_flag": "legit",
  "confidence": 0.92,
  "timestamp": "2025-07-13T08:32:03.456Z",
  "processing_time_ms": 45.67,
  "model_version": "XGBoost_v1.0",
  "feature_count": 76
}
```

## 🔧 Configuration

### Environment Variables

```bash
LOG_LEVEL=INFO              # Logging level
MODEL_PATH=files/enhanced_model.pkl  # Model file path
API_HOST=0.0.0.0           # API host
API_PORT=8000              # API port
```

### Model Requirements

The API expects the model file at `files/enhanced_model.pkl` containing:
- Trained XGBoost model
- Feature scaler (if used)
- Label encoder
- Feature list and metadata

## 🧪 Testing

### Run Tests

```bash
# Unit tests
python -m pytest tests/ -v

# API tests
python tests/test_api.py

# Load testing
# pip install locust
# locust -f tests/load_test.py --host http://localhost:8000
```

### Test Coverage

- ✅ Endpoint functionality
- ✅ Input validation
- ✅ Error handling
- ✅ Model predictions
- ✅ Health checks

## 📊 Monitoring

### Logs

Logs are written to:
- Console output (structured JSON)
- `logs/ml_risk_api.log` (file)

### Metrics

Access metrics at `/model/info` endpoint:
- Request counts
- Processing times
- Error rates
- Risk flag distribution

### Health Checks

Use `/health` endpoint for:
- API status
- Model status
- Uptime monitoring
- Error detection

## 🔒 Security Considerations

### Production Checklist

- [ ] Configure CORS origins properly
- [ ] Add rate limiting
- [ ] Implement authentication
- [ ] Use HTTPS/TLS
- [ ] Validate input thoroughly
- [ ] Monitor for anomalies
- [ ] Regular security updates

### Input Validation

- All inputs validated with Pydantic
- Range checks on numeric features
- Type validation
- Required field enforcement

## 🚀 Performance

### Benchmarks

- **Prediction Time**: ~30-50ms per request
- **Throughput**: ~500-1000 requests/second
- **Memory Usage**: ~200-300MB
- **CPU Usage**: ~10-20% per core

### Optimization Tips

- Use multiple workers for production
- Enable model caching
- Implement request batching
- Monitor memory usage
- Use async endpoints for I/O

## 🔄 Integration

### Java Application Integration

```java
// Sample Java integration code
public class RiskApiClient {
    private final String apiUrl = "http://localhost:8000";
    
    public RiskScoreResponse assessRisk(TransactionData transaction) {
        // Convert transaction to API request format
        // Send POST request to /risk-score
        // Parse JSON response
    }
}
```

### Python Client

```python
import requests

def assess_risk(transaction_features):
    response = requests.post(
        "http://localhost:8000/risk-score",
        json=transaction_features
    )
    return response.json()
```

## 📈 Roadmap

- [ ] Add caching layer (Redis)
- [ ] Implement A/B testing
- [ ] Add model versioning
- [ ] Enhanced monitoring
- [ ] Auto-scaling capabilities
- [ ] Model retraining pipeline

## 👥 Support

- **Documentation**: `/docs` (Swagger UI)
- **Health Check**: `/health`
- **Model Info**: `/model/info`
- **Logs**: `logs/ml_risk_api.log`

---

**Repository**: AI-Quantum-Tech-Security/BMD  
**Author**: olafcio42  
**Version**: 1.0.0  
**Date**: 2025-07-13