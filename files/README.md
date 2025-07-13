# Enhanced ML Risk API - Production Ready

A consolidated, production-ready FastAPI microservice for real-time behavioral risk assessment with comprehensive HTTPS, CORS, security, and monitoring capabilities.

## 🚀 Features

### Core Functionality
- **Single File Implementation**: All functionality consolidated into `files/risk_api.py`
- **FastAPI Framework**: Automatic OpenAPI documentation with Swagger UI and ReDoc
- **Graceful Fallback**: Works with or without ML dependencies, providing mock predictions when needed
- **Comprehensive Validation**: Request/response validation with Pydantic models

### Security & HTTPS
- **SSL/TLS Support**: Full HTTPS configuration with automatic self-signed certificate generation
- **Security Headers**: HSTS, CSP, X-Frame-Options, and other security headers
- **API Key Authentication**: Optional Bearer token authentication
- **Rate Limiting**: Configurable per-IP rate limiting
- **Input Sanitization**: Comprehensive request validation and error handling

### CORS & Networking
- **Advanced CORS**: Environment-specific origin configuration
- **Trusted Hosts**: Production-ready host validation
- **Preflight Support**: Proper handling of OPTIONS requests
- **Custom Headers**: Processing time and request ID tracking

### Monitoring & Logging
- **Request/Response Logging**: Comprehensive request tracking with structured logging
- **Performance Metrics**: Real-time API performance monitoring
- **Health Checks**: Detailed system health endpoints with metrics
- **Error Tracking**: Centralized error logging and alerting

### Model Integration
- **Multiple Model Formats**: Support for pickle, joblib, and other formats
- **Version Tracking**: Model metadata and performance tracking
- **Batch Processing**: Efficient batch prediction capabilities
- **Graceful Degradation**: Intelligent fallback when models are unavailable

## 📁 File Structure

```
files/
├── risk_api.py          # Main consolidated API file
└── certs/               # SSL certificates (auto-generated)
    ├── cert.pem
    └── key.pem
logs/
└── risk_api.log         # Application logs
```

## 🛠️ Installation & Dependencies

### Required Dependencies (for full functionality)
```bash
pip install fastapi uvicorn[standard] pandas numpy scikit-learn xgboost joblib pydantic
```

### Optional Dependencies
```bash
pip install psutil structlog  # For enhanced monitoring
```

### No-Dependency Mode
The API includes a dependency-free mode that provides basic functionality even without FastAPI or ML libraries installed.

## 🚀 Quick Start

### 1. Development Mode (HTTP)
```bash
python files/risk_api.py
```
Access at: http://localhost:8443

### 2. Production Mode (HTTPS)
```bash
RISK_API_SSL_ENABLED=true python files/risk_api.py
```
Access at: https://localhost:8443

### 3. Custom Configuration
```bash
RISK_API_PORT=8443 \
RISK_API_SSL_ENABLED=true \
RISK_API_CORS_ORIGINS=https://app.example.com,https://admin.example.com \
RISK_API_RATE_LIMIT=200 \
python files/risk_api.py
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_API_HOST` | `0.0.0.0` | Server host address |
| `RISK_API_PORT` | `8443` | Server port |
| `RISK_API_SSL_ENABLED` | `false` | Enable HTTPS/SSL |
| `RISK_API_SSL_CERT_PATH` | `./certs/cert.pem` | SSL certificate path |
| `RISK_API_SSL_KEY_PATH` | `./certs/key.pem` | SSL private key path |
| `RISK_API_CORS_ORIGINS` | `http://localhost:3000,https://localhost:3000` | Allowed CORS origins |
| `RISK_API_LOG_LEVEL` | `INFO` | Logging level |
| `RISK_API_MODEL_PATH` | `files/enhanced_model.pkl` | ML model file path |
| `RISK_API_RATE_LIMIT` | `100` | Requests per window |
| `RISK_API_RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |
| `RISK_API_API_KEY` | `None` | Optional API key for authentication |
| `RISK_API_DEV_MODE` | `true` | Development mode (affects CORS and security) |

### Production Deployment Example
```bash
RISK_API_DEV_MODE=false \
RISK_API_SSL_ENABLED=true \
RISK_API_SSL_CERT_PATH=/etc/ssl/certs/api.crt \
RISK_API_SSL_KEY_PATH=/etc/ssl/private/api.key \
RISK_API_CORS_ORIGINS=https://yourdomain.com \
RISK_API_API_KEY=your-secret-production-key \
python files/risk_api.py
```

## 📡 API Endpoints

### Core Endpoints

#### `GET /`
API information and status
```json
{
  "name": "ML Risk API - Enhanced Behavioral Authentication",
  "version": "1.0.0",
  "status": "operational",
  "ssl_enabled": true,
  "model_status": "loaded",
  "endpoints": {...},
  "features": [...]
}
```

#### `GET /health`
Comprehensive health check with system metrics
```json
{
  "status": "healthy",
  "timestamp": "2025-07-13T11:14:00Z",
  "model_status": "loaded",
  "api_version": "1.0.0",
  "uptime_seconds": 3600.5,
  "ssl_enabled": true,
  "cors_origins": ["https://yourdomain.com"]
}
```

#### `GET /model/info`
Model metadata and performance information
```json
{
  "model_type": "XGBoost",
  "model_version": "1.0",
  "training_date": "2025-07-13T08:00:00Z",
  "feature_count": 76,
  "classes": ["legit", "suspicious", "fraud"],
  "training_metrics": {
    "f1_score": 0.9778,
    "roc_auc_macro": 0.9856,
    "cv_mean": 0.9976
  },
  "status": "loaded"
}
```

### Prediction Endpoints

#### `POST /risk-score`
Single transaction risk assessment
```bash
curl -X POST "https://localhost:8443/risk-score" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \
     -d '{
       "avg_tx_amount": 500.0,
       "tx_hour": 14,
       "device_change_freq": 0.1,
       "location_change_freq": 0.05,
       "is_new_device": 0,
       "transaction_count_24h": 3,
       "time_since_last_tx": 2.0,
       "tx_amount_to_balance_ratio": 0.2,
       "ip_address_reputation": 0.85,
       "is_weekend": 0,
       "transaction_velocity_10min": 1,
       "country_change_flag": 0
     }'
```

Response:
```json
{
  "request_id": "req_1720868040000",
  "risk_score": 0.75,
  "risk_flag": "suspicious",
  "confidence": 0.85,
  "timestamp": "2025-07-13T11:14:00Z",
  "processing_time_ms": 42.5,
  "model_version": "XGBoost_v1.0",
  "feature_count": 76
}
```

#### `POST /risk-score/batch`
Batch risk assessment (up to 100 requests)
```json
[
  {
    "avg_tx_amount": 100.0,
    "tx_hour": 10,
    ...
  },
  {
    "avg_tx_amount": 500.0,
    "tx_hour": 22,
    ...
  }
]
```

### Monitoring Endpoints

#### `GET /metrics`
Performance metrics and statistics
```json
{
  "api_metrics": {
    "uptime_seconds": 7200.5,
    "total_requests": 1543,
    "error_rate": 0.023,
    "average_response_time_ms": 45.2,
    "requests_per_second": 0.21,
    "risk_flag_distribution": {
      "legit": 1200,
      "suspicious": 280,
      "fraud": 63
    }
  },
  "model_metrics": {
    "prediction_count": 1543,
    "error_count": 2,
    "average_prediction_time_ms": 12.3
  }
}
```

### Documentation Endpoints

- `GET /docs` - Swagger UI interactive documentation
- `GET /redoc` - ReDoc documentation

## 🔧 Request Format

### Required Fields
```json
{
  "avg_tx_amount": 500.0,        // Average transaction amount
  "tx_hour": 14,                 // Hour of transaction (0-23)
  "device_change_freq": 0.1,     // Device change frequency (0-1)
  "location_change_freq": 0.05,  // Location change frequency (0-1)
  "is_new_device": 0,            // New device flag (0/1)
  "transaction_count_24h": 3,    // 24h transaction count
  "time_since_last_tx": 2.0,     // Hours since last transaction
  "tx_amount_to_balance_ratio": 0.2,  // Transaction to balance ratio
  "ip_address_reputation": 0.85, // IP reputation score (0-1)
  "is_weekend": 0,               // Weekend flag (0/1)
  "transaction_velocity_10min": 1, // 10min transaction velocity
  "country_change_flag": 0       // Country change flag (0/1)
}
```

### Optional Enhanced Fields
```json
{
  "session_duration": 15.5,
  "tx_amount": 750.0,
  "account_age_days": 365,
  "typing_pattern_similarity": 0.92,
  "mouse_movement_similarity": 0.88,
  "merchant_id": "MERCHANT_001",
  "device_type": "mobile",
  "location": "New York"
}
```

## 🛡️ Security Features

### Rate Limiting
- Configurable per-IP rate limiting
- Default: 100 requests per 60 seconds
- Returns HTTP 429 when exceeded

### API Key Authentication
```bash
# Optional Bearer token authentication
curl -H "Authorization: Bearer your-secret-key" ...
```

### Security Headers
- `Strict-Transport-Security` (HTTPS only)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy`

### Input Validation
- Comprehensive Pydantic model validation
- Range checking for numeric fields
- Type validation for all inputs
- Sanitized error messages (no sensitive data leakage)

## 📊 Monitoring & Logging

### Structured Logging
- Request/response logging with timing
- Error tracking with stack traces
- Performance metrics logging
- Configurable log levels

### Health Monitoring
- API uptime tracking
- Request/response metrics
- Error rate monitoring
- Model performance tracking

### Log Files
```
logs/risk_api.log    # Main application log
```

## 🔄 Model Integration

### Supported Model Formats
- Joblib (.pkl files)
- Pickle format
- Custom model packages

### Model Package Structure
```python
{
    'model': trained_model,           # The actual ML model
    'scaler': preprocessing_scaler,   # Optional feature scaler
    'features': feature_list,         # List of feature names
    'model_type': 'XGBoost',         # Model type identifier
    'training_date': '2025-07-13',   # Training timestamp
    'training_metrics': {...}        # Performance metrics
}
```

### Graceful Fallback
When ML models are unavailable, the API provides intelligent mock predictions based on:
- Transaction amount patterns
- Time-based risk factors
- Behavioral indicators
- Network reputation signals

## 🚨 Error Handling

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (invalid API key)
- `422` - Validation Error
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
- `503` - Service Unavailable (model loading issues)

### Error Response Format
```json
{
  "error": "ValidationError",
  "message": "tx_hour must be between 0 and 23",
  "request_id": "req_1720868040000",
  "timestamp": "2025-07-13T11:14:00Z"
}
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_risk_api.py
```

The test suite validates:
- Core functionality without dependencies
- Configuration management
- Rate limiting
- Model service operations
- Mock predictions
- API metrics
- Pydantic models

## 📈 Performance

### Benchmarks (Mock Mode)
- **Response Time**: ~45ms average
- **Throughput**: ~200 requests/second
- **Memory Usage**: ~50MB base
- **Startup Time**: ~2 seconds

### Production Recommendations
- Use a reverse proxy (nginx) for SSL termination
- Deploy with multiple workers (gunicorn/uvicorn)
- Implement external rate limiting (Redis)
- Use external monitoring (Prometheus/Grafana)
- Configure log rotation

## 🔧 Troubleshooting

### Common Issues

#### 1. SSL Certificate Issues
```bash
# Manually generate certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

#### 2. Port Already in Use
```bash
# Check what's using the port
lsof -i :8443

# Use different port
RISK_API_PORT=9443 python files/risk_api.py
```

#### 3. Model Loading Issues
Check the logs for specific error messages:
```bash
tail -f logs/risk_api.log
```

#### 4. CORS Issues
Update CORS origins:
```bash
RISK_API_CORS_ORIGINS=https://your-frontend-domain.com python files/risk_api.py
```

## 📞 Support

For issues and support:
1. Check the logs in `logs/risk_api.log`
2. Run the test suite: `python test_risk_api.py`
3. Review the API health endpoint: `GET /health`
4. Check the metrics endpoint: `GET /metrics`

## 📄 License

This project is part of the AI-Quantum-Tech-Security/BMD repository.