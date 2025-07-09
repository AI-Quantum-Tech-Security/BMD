from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the model
try:
    model_package = joblib.load('model.pkl')
    
    # Handle both dictionary format and direct model format
    if isinstance(model_package, dict):
        model = model_package['model']
        feature_names = model_package.get('features', model_package.get('feature_names'))
        scaler = model_package.get('scaler')
        model_type = model_package.get('model_type', 'Unknown')
    else:
        # Direct model object (legacy format)
        model = model_package
        feature_names = None
        scaler = None
        model_type = type(model).__name__
    
    logger.info("Model loaded successfully")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Features count: {len(feature_names) if feature_names else 'Using API schema'}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

app = FastAPI(
    title="Risk Scoring API",
    description="API for behavioral risk scoring",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionFeatures(BaseModel):
    avg_tx_amount: float
    device_change_freq: float
    tx_hour: int
    location_change_freq: float
    is_new_device: int
    transaction_count_24h: int
    time_since_last_tx: float
    tx_amount_to_balance_ratio: float
    ip_address_reputation: float
    is_weekend: int
    transaction_velocity_10min: int
    country_change_flag: int


class RiskResponse(BaseModel):
    risk_score: float
    risk_flag: str
    timestamp: str
    request_id: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Risk Scoring API",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.post("/risk-score", response_model=RiskResponse)
async def predict_risk(features: TransactionFeatures):
    request_id = str(uuid.uuid4())
    logger.info(f"Request received - ID: {request_id}")
    
    # Log input features for debugging
    input_features = {
        "avg_tx_amount": features.avg_tx_amount,
        "device_change_freq": features.device_change_freq,
        "tx_hour": features.tx_hour,
        "location_change_freq": features.location_change_freq,
        "is_new_device": features.is_new_device,
        "transaction_count_24h": features.transaction_count_24h,
        "time_since_last_tx": features.time_since_last_tx,
        "tx_amount_to_balance_ratio": features.tx_amount_to_balance_ratio,
        "ip_address_reputation": features.ip_address_reputation,
        "is_weekend": features.is_weekend,
        "transaction_velocity_10min": features.transaction_velocity_10min,
        "country_change_flag": features.country_change_flag
    }
    logger.info(f"Input features - ID: {request_id} - {input_features}")

    try:
        # Convert features to array
        feature_array = np.array([[
            features.avg_tx_amount,
            features.device_change_freq,
            features.tx_hour,
            features.location_change_freq,
            features.is_new_device,
            features.transaction_count_24h,
            features.time_since_last_tx,
            features.tx_amount_to_balance_ratio,
            features.ip_address_reputation,
            features.is_weekend,
            features.transaction_velocity_10min,
            features.country_change_flag
        ]])

        logger.debug(f"Feature array shape - ID: {request_id} - {feature_array.shape}")

        # Scale features if scaler exists
        if scaler:
            feature_array_original = feature_array.copy()
            feature_array = scaler.transform(feature_array)
            logger.debug(f"Features scaled - ID: {request_id} - Original range: [{feature_array_original.min():.3f}, {feature_array_original.max():.3f}], Scaled range: [{feature_array.min():.3f}, {feature_array.max():.3f}]")

        # Make prediction
        risk_score = model.predict_proba(feature_array)[0]
        risk_flag = model.predict(feature_array)[0]

        # Prepare response
        response = {
            "risk_score": float(max(risk_score)),
            "risk_flag": str(risk_flag),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id
        }

        # Log output for debugging
        logger.info(f"Prediction output - ID: {request_id} - Risk Score: {response['risk_score']:.4f}, Risk Flag: {response['risk_flag']}, Probability Distribution: {risk_score}")
        
        return response

    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("risk_api:app", host="0.0.0.0", port=8000, reload=True,
                ssl_keyfile="key.pem", ssl_certfile="cert.pem")