# ML API Integration Guide

**Author:** olafcio42  
**Date:** 2025-07-13  
**Version:** 1.0  
**Repository:** AI-Quantum-Tech-Security/BMD

## Overview

This guide provides comprehensive instructions for integrating with the BMD ML Risk API, a behavioral authentication microservice that provides real-time fraud detection capabilities using machine learning.

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Authentication & Security](#authentication--security)
4. [Request/Response Format](#requestresponse-format)
5. [Java Integration](#java-integration)
6. [Error Handling](#error-handling)
7. [Testing & Debugging](#testing--debugging)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+ with required dependencies
- SSL certificate (optional - self-signed generated automatically)
- Network access to API server

### Installation & Setup

1. **Install Dependencies:**
```bash
pip install flask flask-cors cryptography joblib pandas numpy scikit-learn xgboost
```

2. **Start the API Server:**
```bash
python risk_api.py
```

3. **Verify Installation:**
```bash
curl -k https://localhost:5000/
```

Expected response:
```json
{
  "status": "healthy",
  "service": "ML Risk API - Behavioral Authentication",
  "version": "1.0.0",
  "model_loaded": true
}
```

## API Endpoints

### Base URL
- **HTTPS:** `https://localhost:5000` (recommended)
- **HTTP:** `http://localhost:5000` (fallback)

### 1. Health Check
**Endpoint:** `GET /`

**Description:** Check API status and uptime

**Response:**
```json
{
  "status": "healthy",
  "service": "ML Risk API - Behavioral Authentication",
  "version": "1.0.0",
  "uptime_seconds": 3600.45,
  "timestamp": "2025-07-13T14:36:38.123456+00:00",
  "model_loaded": true,
  "endpoints": {
    "health": "/",
    "predict": "/predict",
    "model_info": "/model-info"
  }
}
```

### 2. Risk Prediction
**Endpoint:** `POST /predict`

**Description:** Analyze behavioral features and return risk assessment

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [
    0.5,    // avg_tx_amount (normalized)
    0.2,    // device_change_freq
    0.8,    // tx_hour (normalized)
    0.1,    // location_change_freq
    0.0,    // is_new_device
    0.3,    // velocity_last_hour
    0.6,    // amount_deviation
    0.4,    // time_since_last_tx
    0.2,    // geo_velocity
    0.7     // user_behavior_score
  ]
}
```

**Response:**
```json
{
  "prediction": "legit",
  "confidence": 0.8745,
  "risk_score": 0.1255,
  "probabilities": {
    "legit": 0.8745,
    "suspicious": 0.1024,
    "fraud": 0.0231
  },
  "mock_mode": false,
  "timestamp": "2025-07-13T14:36:38.123456+00:00",
  "feature_count": 10,
  "api_version": "1.0.0"
}
```

### 3. Model Information
**Endpoint:** `GET /model-info`

**Description:** Get model metadata and training information

**Response:**
```json
{
  "model_loaded": true,
  "model_type": "XGBoost",
  "training_date": "2025-07-13",
  "feature_count": 10,
  "features": [
    "avg_tx_amount",
    "device_change_freq",
    "tx_hour",
    "location_change_freq",
    "is_new_device",
    "velocity_last_hour",
    "amount_deviation",
    "time_since_last_tx",
    "geo_velocity",
    "user_behavior_score"
  ],
  "classes": ["legit", "suspicious", "fraud"],
  "training_metrics": {
    "f1_score": 0.9778,
    "accuracy": 0.9701,
    "precision": 0.9734,
    "recall": 0.9823
  },
  "timestamp": "2025-07-13T14:36:38.123456+00:00",
  "api_version": "1.0.0"
}
```

## Authentication & Security

### HTTPS Configuration

The API automatically generates self-signed certificates for development. For production:

1. **Place SSL certificates:**
```
ssl/
├── cert.pem
└── key.pem
```

2. **Environment variables:**
```bash
export SSL_CERT_PATH="ssl/cert.pem"
export SSL_KEY_PATH="ssl/key.pem"
```

### CORS Settings

CORS is enabled for all origins by default. For production, configure specific origins:

```python
CORS(app, origins=["https://yourdomain.com", "https://app.yourdomain.com"])
```

## Request/Response Format

### Feature Vector Specification

The API expects exactly **10 numerical features** in the following order:

| Index | Feature Name | Type | Range | Description |
|-------|-------------|------|--------|-------------|
| 0 | avg_tx_amount | float | 0.0-1.0 | Normalized average transaction amount |
| 1 | device_change_freq | float | 0.0-1.0 | Device change frequency |
| 2 | tx_hour | float | 0.0-1.0 | Transaction hour (normalized 0-23) |
| 3 | location_change_freq | float | 0.0-1.0 | Location change frequency |
| 4 | is_new_device | float | 0.0/1.0 | Boolean: new device flag |
| 5 | velocity_last_hour | float | 0.0-1.0 | Transaction velocity in last hour |
| 6 | amount_deviation | float | 0.0-1.0 | Deviation from user's normal amounts |
| 7 | time_since_last_tx | float | 0.0-1.0 | Time since last transaction |
| 8 | geo_velocity | float | 0.0-1.0 | Geographic velocity |
| 9 | user_behavior_score | float | 0.0-1.0 | Historical behavior score |

### Risk Score Interpretation

| Risk Score | Prediction | Action Recommendation |
|------------|------------|----------------------|
| 0.0 - 0.3 | legit | Approve transaction |
| 0.3 - 0.7 | suspicious | Additional verification |
| 0.7 - 1.0 | fraud | Block transaction |

## Java Integration

### 1. Dependencies (Maven)

```xml
<dependencies>
    <dependency>
        <groupId>com.fasterxml.jackson.core</groupId>
        <artifactId>jackson-databind</artifactId>
        <version>2.15.2</version>
    </dependency>
    <dependency>
        <groupId>org.apache.httpcomponents</groupId>
        <artifactId>httpclient</artifactId>
        <version>4.5.14</version>
    </dependency>
</dependencies>
```

### 2. Java Client Implementation

```java
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

public class RiskApiClient {
    private static final String API_BASE_URL = "https://localhost:5000";
    private final CloseableHttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    public RiskApiClient() {
        this.httpClient = HttpClients.createDefault();
        this.objectMapper = new ObjectMapper();
    }
    
    public RiskPrediction predictRisk(double[] features) throws Exception {
        HttpPost request = new HttpPost(API_BASE_URL + "/predict");
        request.setHeader("Content-Type", "application/json");
        
        // Create request body
        PredictRequest requestBody = new PredictRequest(features);
        String jsonBody = objectMapper.writeValueAsString(requestBody);
        request.setEntity(new StringEntity(jsonBody));
        
        // Execute request
        try (var response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            return objectMapper.readValue(responseBody, RiskPrediction.class);
        }
    }
    
    // Request/Response DTOs
    public static class PredictRequest {
        public double[] features;
        
        public PredictRequest(double[] features) {
            this.features = features;
        }
    }
    
    public static class RiskPrediction {
        public String prediction;
        public double confidence;
        public double risk_score;
        public Map<String, Double> probabilities;
        public boolean mock_mode;
        public String timestamp;
        public int feature_count;
        public String api_version;
    }
}
```

### 3. Usage Example

```java
public class ExampleUsage {
    public static void main(String[] args) {
        try {
            RiskApiClient client = new RiskApiClient();
            
            // Synthetic transaction features
            double[] features = {
                0.45,  // avg_tx_amount
                0.12,  // device_change_freq
                0.67,  // tx_hour (16:00 normalized)
                0.05,  // location_change_freq
                0.0,   // is_new_device
                0.23,  // velocity_last_hour
                0.34,  // amount_deviation
                0.56,  // time_since_last_tx
                0.08,  // geo_velocity
                0.78   // user_behavior_score
            };
            
            RiskPrediction result = client.predictRisk(features);
            
            System.out.println("Prediction: " + result.prediction);
            System.out.println("Risk Score: " + result.risk_score);
            System.out.println("Confidence: " + result.confidence);
            
            // Business logic based on risk score
            if (result.risk_score < 0.3) {
                System.out.println("Action: APPROVE");
            } else if (result.risk_score < 0.7) {
                System.out.println("Action: ADDITIONAL_VERIFICATION");
            } else {
                System.out.println("Action: BLOCK");
            }
            
        } catch (Exception e) {
            System.err.println("API call failed: " + e.getMessage());
        }
    }
}
```

## Error Handling

### HTTP Status Codes

| Code | Description | Action |
|------|-------------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check request format |
| 404 | Not Found | Check endpoint URL |
| 500 | Internal Server Error | Retry or contact support |

### Error Response Format

```json
{
  "error": "Bad Request",
  "message": "Features must be a list",
  "timestamp": "2025-07-13T14:36:38.123456+00:00"
}
```

### Common Error Scenarios

1. **Invalid Feature Count:**
```json
{
  "error": "Invalid features",
  "message": "Expected 10 features, got 8"
}
```

2. **Model Not Loaded:**
```json
{
  "prediction": "error",
  "mock_mode": true,
  "reason": "Model file not found"
}
```

3. **Network Timeout:**
```java
// Java retry logic
public RiskPrediction predictRiskWithRetry(double[] features, int maxRetries) {
    for (int i = 0; i < maxRetries; i++) {
        try {
            return predictRisk(features);
        } catch (Exception e) {
            if (i == maxRetries - 1) throw e;
            Thread.sleep(1000 * (i + 1)); // Exponential backoff
        }
    }
}
```

## Testing & Debugging

### 1. Synthetic Test Data

```bash
# Test legitimate transaction
curl -k -X POST https://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.3, 0.1, 0.5, 0.05, 0.0, 0.2, 0.1, 0.4, 0.02, 0.8]
  }'

# Test suspicious transaction
curl -k -X POST https://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.8, 0.6, 0.9, 0.7, 1.0, 0.9, 0.8, 0.1, 0.9, 0.2]
  }'
```

### 2. Logging & Monitoring

The API includes comprehensive logging for debugging:

```python
# Example log output
INFO:__main__:Prediction request: features=[0.3, 0.1, 0.5, ...]
INFO:__main__:Model prediction: legit (confidence: 0.87)
INFO:__main__:Response sent: risk_score=0.13
```

### 3. Performance Testing

```bash
# Stress test with concurrent requests
for i in {1..100}; do
  curl -k -X POST https://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [0.5,0.2,0.8,0.1,0.0,0.3,0.6,0.4,0.2,0.7]}' &
done
wait
```

## Performance Considerations

### Response Times
- **Typical:** 50-100ms per prediction
- **Peak load:** 200-500ms with concurrent requests
- **Timeout recommended:** 5 seconds

### Rate Limiting
Consider implementing rate limiting for production:

```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per minute")
def predict():
    # ... existing code
```

### Caching Strategy
For high-frequency users, implement feature caching:

```java
// Java example with Redis
public class CachedRiskClient {
    private final RiskApiClient apiClient;
    private final RedisTemplate<String, RiskPrediction> cache;
    
    public RiskPrediction predictWithCache(String userId, double[] features) {
        String cacheKey = generateCacheKey(userId, features);
        RiskPrediction cached = cache.opsForValue().get(cacheKey);
        
        if (cached != null && !isExpired(cached)) {
            return cached;
        }
        
        RiskPrediction fresh = apiClient.predictRisk(features);
        cache.opsForValue().set(cacheKey, fresh, Duration.ofMinutes(5));
        return fresh;
    }
}
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors:**
```bash
# For development, use -k flag to ignore SSL errors
curl -k https://localhost:5000/

# Or configure Java to trust self-signed certificates
System.setProperty("javax.net.ssl.trustStore", "path/to/truststore");
```

2. **Model Loading Issues:**
```bash
# Check model file exists
ls -la files/enhanced_model.pkl

# Check Python dependencies
pip list | grep -E "(joblib|pandas|numpy|scikit-learn|xgboost)"
```

3. **Port Conflicts:**
```bash
# Change port if 5000 is occupied
export PORT=8080
python risk_api.py
```

4. **Memory Issues:**
```bash
# Monitor memory usage
ps aux | grep risk_api
# Restart if memory usage > 1GB
```

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
python risk_api.py
```

### Health Monitoring

Implement health checks in your application:

```java
public boolean isApiHealthy() {
    try {
        HttpGet request = new HttpGet(API_BASE_URL + "/");
        var response = httpClient.execute(request);
        return response.getStatusLine().getStatusCode() == 200;
    } catch (Exception e) {
        return false;
    }
}
```

## Production Deployment

### Docker Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY risk_api.py .
COPY files/ ./files/

EXPOSE 5000
CMD ["python", "risk_api.py"]
```

### Environment Variables

```bash
# Production settings
export HOST=0.0.0.0
export PORT=5000
export DEBUG=false
export SSL_CERT_PATH=/etc/ssl/certs/api.crt
export SSL_KEY_PATH=/etc/ssl/private/api.key
```

### Load Balancer Configuration

```nginx
upstream risk_api {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass https://risk_api;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Support

For issues and questions:
- **Repository:** [AI-Quantum-Tech-Security/BMD](https://github.com/AI-Quantum-Tech-Security/BMD)
- **Author:** olafcio42
- **Documentation:** Check `README.md` and API docs at `/docs`

---

**Last Updated:** 2025-07-13  
**Version:** 1.0.0