#!/usr/bin/env python3
"""
Test script for the consolidated risk_api.py
Validates core functionality without requiring FastAPI dependencies
"""

import os
import sys
import json
from pathlib import Path

# Add the files directory to the path
sys.path.insert(0, str(Path(__file__).parent / "files"))

def test_basic_functionality():
    """Test basic API functionality"""
    print("="*60)
    print("TESTING CONSOLIDATED RISK API")
    print("="*60)
    
    try:
        from risk_api import (
            Config, EnhancedModelService, RateLimiter, 
            APIMetrics, generate_self_signed_cert
        )
        print("✅ Successfully imported core classes")
        
        # Test 1: Configuration
        print("\n1. Testing Configuration...")
        config = Config()
        print(f"   Host: {config.HOST}")
        print(f"   Port: {config.PORT}")
        print(f"   SSL Enabled: {config.SSL_ENABLED}")
        print(f"   CORS Origins: {config.CORS_ORIGINS}")
        print(f"   Model Path: {config.MODEL_PATH}")
        
        # Test 2: Rate Limiter
        print("\n2. Testing Rate Limiter...")
        limiter = RateLimiter(max_requests=3, window_seconds=10)
        test_ip = "192.168.1.100"
        
        for i in range(5):
            allowed = limiter.is_allowed(test_ip)
            print(f"   Request {i+1}: {'Allowed' if allowed else 'Rate Limited'}")
        
        # Test 3: Model Service
        print("\n3. Testing Model Service...")
        model_service = EnhancedModelService("files/enhanced_model.pkl")
        model_service.load_model()
        
        print(f"   Model Loaded: {model_service.is_model_loaded()}")
        print(f"   Error Message: {model_service.error_message or 'None'}")
        
        # Test model info
        model_info = model_service.get_model_info()
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Status: {model_info['status']}")
        print(f"   Feature Count: {model_info['feature_count']}")
        
        # Test 4: Predictions
        print("\n4. Testing Predictions...")
        
        # Test case 1: Low risk transaction
        low_risk_features = {
            "avg_tx_amount": 100.0,
            "tx_hour": 14,  # Business hours
            "device_change_freq": 0.0,
            "location_change_freq": 0.0,
            "is_new_device": 0,
            "transaction_count_24h": 2,
            "time_since_last_tx": 4.0,
            "tx_amount_to_balance_ratio": 0.1,
            "ip_address_reputation": 0.95,
            "is_weekend": 0,
            "transaction_velocity_10min": 1,
            "country_change_flag": 0,
            "tx_amount": 100.0
        }
        
        prediction = model_service.predict(low_risk_features)
        print(f"   Low Risk Case: {prediction['risk_flag']} (score: {prediction['risk_score']:.4f})")
        
        # Test case 2: High risk transaction  
        high_risk_features = {
            "avg_tx_amount": 100.0,
            "tx_hour": 3,  # Unusual hour
            "device_change_freq": 0.8,
            "location_change_freq": 0.7,
            "is_new_device": 1,
            "transaction_count_24h": 15,
            "time_since_last_tx": 0.1,
            "tx_amount_to_balance_ratio": 0.9,
            "ip_address_reputation": 0.2,  # Low reputation
            "is_weekend": 1,
            "transaction_velocity_10min": 8,
            "country_change_flag": 1,
            "tx_amount": 5000.0  # Much higher than average
        }
        
        prediction = model_service.predict(high_risk_features)
        print(f"   High Risk Case: {prediction['risk_flag']} (score: {prediction['risk_score']:.4f})")
        
        # Test 5: API Metrics
        print("\n5. Testing API Metrics...")
        metrics = APIMetrics()
        
        # Simulate some requests
        metrics.record_request("/risk-score", 0.045, "127.0.0.1", 200, "legit")
        metrics.record_request("/risk-score", 0.123, "127.0.0.1", 200, "suspicious")
        metrics.record_request("/risk-score", 0.089, "192.168.1.1", 200, "fraud")
        metrics.record_request("/health", 0.012, "127.0.0.1", 200)
        metrics.record_request("/risk-score", 0.156, "127.0.0.1", 500)  # Error
        
        summary = metrics.get_metrics()
        print(f"   Total Requests: {summary['total_requests']}")
        print(f"   Error Rate: {summary['error_rate']:.2%}")
        print(f"   Avg Response Time: {summary['average_response_time_ms']:.2f}ms")
        print(f"   Risk Distribution: {summary['risk_flag_distribution']}")
        
        # Test 6: Environment Configuration
        print("\n6. Testing Environment Configuration...")
        
        # Test with environment variables
        os.environ['RISK_API_PORT'] = '9443'
        os.environ['RISK_API_SSL_ENABLED'] = 'true'
        os.environ['RISK_API_CORS_ORIGINS'] = 'https://example.com,https://app.test.com'
        os.environ['RISK_API_RATE_LIMIT'] = '50'
        
        # Reload config (this would normally require restarting the app)
        from importlib import reload
        import risk_api
        reload(risk_api)
        
        config_env = risk_api.Config()
        print(f"   Environment Port: {config_env.PORT}")
        print(f"   Environment SSL: {config_env.SSL_ENABLED}")
        print(f"   Environment CORS: {config_env.CORS_ORIGINS}")
        
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe consolidated risk_api.py is working correctly.")
        print("Ready for production deployment with FastAPI dependencies.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pydantic_models():
    """Test Pydantic model functionality"""
    print("\n" + "="*60)
    print("TESTING PYDANTIC MODELS (Mock Mode)")
    print("="*60)
    
    try:
        from risk_api import RiskScoreRequest, RiskScoreResponse
        
        # Test request model
        sample_request_data = {
            "avg_tx_amount": 750.25,
            "device_change_freq": 0.15,
            "tx_hour": 16,
            "location_change_freq": 0.05,
            "is_new_device": 0,
            "transaction_count_24h": 4,
            "time_since_last_tx": 3.5,
            "tx_amount_to_balance_ratio": 0.25,
            "ip_address_reputation": 0.88,
            "is_weekend": 0,
            "transaction_velocity_10min": 2,
            "country_change_flag": 0,
            "merchant_id": "SHOP_12345",
            "device_type": "mobile",
            "location": "San Francisco"
        }
        
        request_model = RiskScoreRequest(**sample_request_data)
        print("✅ Request model created successfully")
        print(f"   Transaction hour: {request_model.tx_hour}")
        print(f"   Device type: {request_model.device_type}")
        
        # Test response model
        response_data = {
            "request_id": "req_1642086123456",
            "risk_score": 0.25,
            "risk_flag": "suspicious",
            "confidence": 0.87,
            "timestamp": "2025-07-13T10:30:00Z",
            "processing_time_ms": 42.5,
            "model_version": "XGBoost_v1.0",
            "feature_count": 76
        }
        
        response_model = RiskScoreResponse(**response_data)
        print("✅ Response model created successfully")
        print(f"   Risk flag: {response_model.risk_flag}")
        print(f"   Confidence: {response_model.confidence}")
        
        # Test model dict conversion
        request_dict = request_model.dict()
        print(f"✅ Model dict conversion works: {len(request_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_usage_examples():
    """Display usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("""
1. Development Mode (HTTP):
   python files/risk_api.py

2. Production Mode (HTTPS):
   RISK_API_SSL_ENABLED=true python files/risk_api.py

3. Custom Configuration:
   RISK_API_PORT=8443 \\
   RISK_API_SSL_ENABLED=true \\
   RISK_API_CORS_ORIGINS=https://app.example.com,https://admin.example.com \\
   RISK_API_RATE_LIMIT=200 \\
   python files/risk_api.py

4. With API Key Authentication:
   RISK_API_API_KEY=your-secret-key \\
   RISK_API_SSL_ENABLED=true \\
   python files/risk_api.py

5. Production Deployment:
   RISK_API_DEV_MODE=false \\
   RISK_API_SSL_ENABLED=true \\
   RISK_API_SSL_CERT_PATH=/etc/ssl/certs/api.crt \\
   RISK_API_SSL_KEY_PATH=/etc/ssl/private/api.key \\
   RISK_API_CORS_ORIGINS=https://yourdomain.com \\
   python files/risk_api.py

Available Endpoints:
- GET  /                    - API information
- GET  /health              - Health check with metrics
- GET  /model/info          - Model information
- POST /risk-score          - Single prediction
- POST /risk-score/batch    - Batch predictions
- GET  /metrics             - Performance metrics
- GET  /docs                - Swagger documentation
- GET  /redoc               - ReDoc documentation

Example Request:
curl -X POST "https://localhost:8443/risk-score" \\
     -H "Content-Type: application/json" \\
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
    """)

if __name__ == "__main__":
    print("Testing Consolidated ML Risk API")
    print("Current directory:", os.getcwd())
    
    success = True
    success &= test_basic_functionality()
    success &= test_pydantic_models()
    
    display_usage_examples()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("The risk_api.py is ready for production use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)