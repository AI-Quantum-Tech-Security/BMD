# ML API Integration Guide

This guide provides comprehensive instructions for integrating with the Risk Scoring API, including examples for various programming languages and tools.

## Table of Contents
- [API Overview](#api-overview)
- [Endpoints](#endpoints)
- [Authentication](#authentication)
- [Request/Response Format](#requestresponse-format)
- [Integration Examples](#integration-examples)
- [Java Integration](#java-integration)
- [Error Handling](#error-handling)
- [Testing](#testing)

## API Overview

The Risk Scoring API is a FastAPI-based microservice that provides behavioral risk assessment for financial transactions. It uses machine learning to analyze transaction features and returns risk scores and flags.

**Base URL:** `https://localhost:8000` (HTTPS) or `http://localhost:8000` (HTTP)
**API Documentation:** `/docs` (Swagger UI)
**OpenAPI Schema:** `/openapi.json`

### Key Features
- Real-time risk scoring
- Enhanced logging for debugging
- CORS support for web integration
- HTTPS support for secure communication
- Comprehensive input validation
- Request tracking with unique IDs

## Endpoints

### Health Check
```
GET /
```
Returns API status and basic information.

**Response:**
```json
{
    "message": "Risk Scoring API",
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2025-07-09 14:53:45",
    "docs": "/docs",
    "openapi": "/openapi.json"
}
```

### Risk Score Prediction
```
POST /risk-score
```
Analyzes transaction features and returns risk assessment.

## Request/Response Format

### Request Schema
The API expects 12 behavioral features as defined in the feature schema:

```json
{
    "avg_tx_amount": 500.0,
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
    "country_change_flag": 0
}
```

### Feature Descriptions
| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| avg_tx_amount | float | Average transaction amount for the user | [20, 10000] |
| device_change_freq | float | Frequency of device changes for the user | [0, 1] |
| tx_hour | int | Hour of the transaction (0-23) | [0, 23] |
| location_change_freq | float | Frequency of location changes for the user | [0, 1] |
| is_new_device | int | Boolean flag for new device (0/1) | 0, 1 |
| transaction_count_24h | int | Number of transactions in last 24h | >=0 |
| time_since_last_tx | float | Hours since previous transaction | >=0 |
| tx_amount_to_balance_ratio | float | Transaction amount to balance ratio | [0, 1] |
| ip_address_reputation | float | IP reputation score (0=bad, 1=good) | [0, 1] |
| is_weekend | int | Boolean flag for weekend transaction (0/1) | 0, 1 |
| transaction_velocity_10min | int | Number of transactions in last 10 minutes | >=0 |
| country_change_flag | int | Boolean flag for country change (0/1) | 0, 1 |

### Response Schema
```json
{
    "risk_score": 0.7381,
    "risk_flag": "legit",
    "timestamp": "2025-07-09 14:53:54",
    "request_id": "a0509387-9a18-49e2-80c8-642a1a2f69b4"
}
```

**Response Fields:**
- `risk_score`: Float between 0-1 (higher = more risky)
- `risk_flag`: One of "legit", "suspicious", "fraud"
- `timestamp`: UTC timestamp of prediction
- `request_id`: Unique identifier for request tracking

## Integration Examples

### cURL Example
```bash
# Basic HTTP request
curl -X POST http://localhost:8000/risk-score \
  -H "Content-Type: application/json" \
  -d '{
    "avg_tx_amount": 500.0,
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
    "country_change_flag": 0
  }'

# HTTPS request (with self-signed cert)
curl -k -X POST https://localhost:8000/risk-score \
  -H "Content-Type: application/json" \
  -d '{...}'  # same JSON payload
```

### Postman Collection
Create a new Postman collection with these requests:

**1. Health Check**
- Method: GET
- URL: `{{baseUrl}}/`
- Headers: None required

**2. Risk Score Prediction**
- Method: POST
- URL: `{{baseUrl}}/risk-score`
- Headers: `Content-Type: application/json`
- Body (raw JSON): Use the request schema above

**Environment Variables:**
- `baseUrl`: `http://localhost:8000` or `https://localhost:8000`

### Python Integration Example
```python
import requests
import json

class RiskScoringClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health status"""
        response = requests.get(f"{self.base_url}/")
        return response.json()
    
    def predict_risk(self, features):
        """Get risk prediction for transaction features"""
        response = requests.post(
            f"{self.base_url}/risk-score",
            headers={"Content-Type": "application/json"},
            json=features
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

# Usage example
client = RiskScoringClient()

# Sample transaction features
features = {
    "avg_tx_amount": 750.0,
    "device_change_freq": 0.02,
    "tx_hour": 10,
    "location_change_freq": 0.05,
    "is_new_device": 0,
    "transaction_count_24h": 3,
    "time_since_last_tx": 1.5,
    "tx_amount_to_balance_ratio": 0.1,
    "ip_address_reputation": 0.9,
    "is_weekend": 0,
    "transaction_velocity_10min": 1,
    "country_change_flag": 0
}

# Get risk prediction
result = client.predict_risk(features)
print(f"Risk Score: {result['risk_score']:.4f}")
print(f"Risk Flag: {result['risk_flag']}")
```

## Java Integration

### Maven Dependencies
Add these dependencies to your `pom.xml`:

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

### Java Client Implementation
```java
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class RiskScoringClient {
    private final String baseUrl;
    private final CloseableHttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    public RiskScoringClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.httpClient = HttpClients.createDefault();
        this.objectMapper = new ObjectMapper();
    }
    
    public Map<String, Object> healthCheck() throws IOException {
        HttpGet request = new HttpGet(baseUrl + "/");
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            HttpEntity entity = response.getEntity();
            String result = EntityUtils.toString(entity);
            return objectMapper.readValue(result, Map.class);
        }
    }
    
    public RiskPrediction predictRisk(TransactionFeatures features) throws IOException {
        HttpPost request = new HttpPost(baseUrl + "/risk-score");
        request.addHeader("Content-Type", "application/json");
        
        String json = objectMapper.writeValueAsString(features);
        StringEntity entity = new StringEntity(json);
        request.setEntity(entity);
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            HttpEntity responseEntity = response.getEntity();
            String result = EntityUtils.toString(responseEntity);
            
            if (response.getStatusLine().getStatusCode() == 200) {
                return objectMapper.readValue(result, RiskPrediction.class);
            } else {
                throw new RuntimeException("API request failed: " + result);
            }
        }
    }
    
    public void close() throws IOException {
        httpClient.close();
    }
}

// Data classes
class TransactionFeatures {
    public double avgTxAmount;
    public double deviceChangeFreq;
    public int txHour;
    public double locationChangeFreq;
    public int isNewDevice;
    public int transactionCount24h;
    public double timeSinceLastTx;
    public double txAmountToBalanceRatio;
    public double ipAddressReputation;
    public int isWeekend;
    public int transactionVelocity10min;
    public int countryChangeFlag;
    
    // Constructor and getters/setters...
}

class RiskPrediction {
    public double riskScore;
    public String riskFlag;
    public String timestamp;
    public String requestId;
    
    // Getters and setters...
}

// Usage example
public class Example {
    public static void main(String[] args) {
        RiskScoringClient client = new RiskScoringClient("http://localhost:8000");
        
        try {
            // Create sample transaction features
            TransactionFeatures features = new TransactionFeatures();
            features.avgTxAmount = 500.0;
            features.deviceChangeFreq = 0.05;
            features.txHour = 14;
            features.locationChangeFreq = 0.15;
            features.isNewDevice = 0;
            features.transactionCount24h = 5;
            features.timeSinceLastTx = 2.5;
            features.txAmountToBalanceRatio = 0.15;
            features.ipAddressReputation = 0.85;
            features.isWeekend = 1;
            features.transactionVelocity10min = 2;
            features.countryChangeFlag = 0;
            
            // Get risk prediction
            RiskPrediction prediction = client.predictRisk(features);
            System.out.println("Risk Score: " + prediction.riskScore);
            System.out.println("Risk Flag: " + prediction.riskFlag);
            
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                client.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
```

### Synthetic Data Integration Example
```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class SyntheticDataProcessor {
    private final RiskScoringClient client;
    
    public SyntheticDataProcessor(String apiBaseUrl) {
        this.client = new RiskScoringClient(apiBaseUrl);
    }
    
    public void processSyntheticDataset(String csvFilePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            boolean isHeader = true;
            
            while ((line = reader.readLine()) != null) {
                if (isHeader) {
                    isHeader = false;
                    continue; // Skip header row
                }
                
                // Parse CSV line and create TransactionFeatures
                TransactionFeatures features = parseCsvLine(line);
                
                // Get risk prediction
                RiskPrediction prediction = client.predictRisk(features);
                
                System.out.printf("Transaction: %s, Risk: %.4f, Flag: %s%n",
                    prediction.requestId, prediction.riskScore, prediction.riskFlag);
            }
        }
    }
    
    private TransactionFeatures parseCsvLine(String csvLine) {
        // Implementation depends on your CSV format
        // Map CSV columns to TransactionFeatures fields
        String[] fields = csvLine.split(",");
        
        TransactionFeatures features = new TransactionFeatures();
        // Map fields to features based on your dataset structure
        // This is a simplified example - adjust based on actual CSV format
        
        return features;
    }
}
```

## Error Handling

### Common HTTP Status Codes
- `200`: Success
- `422`: Validation Error (invalid input format)
- `500`: Internal Server Error

### Example Error Response
```json
{
    "detail": [
        {
            "loc": ["body", "avg_tx_amount"],
            "msg": "field required",
            "type": "value_error.missing"
        }
    ]
}
```

### Python Error Handling Example
```python
try:
    result = client.predict_risk(features)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 422:
        print("Validation error:", e.response.json())
    elif e.response.status_code == 500:
        print("Server error occurred")
    else:
        print(f"HTTP error {e.response.status_code}: {e.response.text}")
except requests.exceptions.ConnectionError:
    print("Could not connect to API server")
```

## Testing

### Unit Testing with Synthetic Data
The API includes test endpoints that work with the synthetic behavioral dataset:

```python
# Load and test with synthetic data
import pandas as pd

# Load the dataset
df = pd.read_csv('files/synthetic_behavioral_dataset.csv')

# Create API-compatible features from dataset row
def map_dataset_to_api_features(row):
    return {
        "avg_tx_amount": float(row['tx_amount']),
        "device_change_freq": 0.05,  # Default value
        "tx_hour": int(row['tx_hour']),
        "location_change_freq": float(row['distance_from_usual_location']) / 1000,
        "is_new_device": int(row['is_new_device']),
        "transaction_count_24h": int(row['transaction_count_24h']),
        "time_since_last_tx": 2.5,  # Default value
        "tx_amount_to_balance_ratio": float(row['tx_amount_to_balance_ratio']),
        "ip_address_reputation": float(row['ip_address_reputation']),
        "is_weekend": int(row['is_weekend']),
        "transaction_velocity_10min": int(row['transaction_velocity_10min']),
        "country_change_flag": int(row['country_change_flag'])
    }

# Test with first 10 rows
for i in range(10):
    features = map_dataset_to_api_features(df.iloc[i])
    result = client.predict_risk(features)
    print(f"Row {i}: Risk={result['risk_score']:.4f}, Flag={result['risk_flag']}")
```

### Performance Testing
```bash
# Install Apache Bench (if not available)
sudo apt-get install apache2-utils

# Run performance test
ab -n 1000 -c 10 -T 'application/json' -p request_body.json http://localhost:8000/risk-score
```

Where `request_body.json` contains:
```json
{
    "avg_tx_amount": 500.0,
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
    "country_change_flag": 0
}
```

## Deployment Considerations

### HTTPS Configuration
The API supports HTTPS with SSL certificates. For production:

1. Obtain valid SSL certificates
2. Update `ssl_keyfile` and `ssl_certfile` paths in `risk_api.py`
3. Configure reverse proxy (nginx/Apache) if needed

### CORS Configuration
Current CORS settings allow all origins (`allow_origins=["*"]`). For production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

### Logging
Enhanced logging includes:
- Request IDs for tracking
- Input feature logging
- Prediction output logging
- Error logging with context

Logs are written to both console and `api.log` file.

---

## Support

For questions or issues with the API integration:
1. Check the Swagger documentation at `/docs`
2. Review the logs for detailed error information
3. Validate input data format against the schema
4. Test with the provided examples

**API Version:** 1.0.0  
**Last Updated:** July 2025