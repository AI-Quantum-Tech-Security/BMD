"""
Response models for ML Risk API
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List


class RiskScoreResponse(BaseModel):
    """Response model for risk score prediction"""

    request_id: str = Field(..., description="Unique request identifier")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score between 0 and 1")
    risk_flag: str = Field(..., description="Risk classification: legit, suspicious, or fraud")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    timestamp: datetime = Field(..., description="Response timestamp in UTC")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used for prediction")
    feature_count: int = Field(..., description="Number of features used in prediction")

    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_1642086123456",
                "risk_score": 0.15,
                "risk_flag": "legit",
                "confidence": 0.92,
                "timestamp": "2025-07-13T08:32:03.456Z",
                "processing_time_ms": 45.67,
                "model_version": "XGBoost_v1.0",
                "feature_count": 76
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(..., description="Overall API health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    model_status: str = Field(..., description="ML model health status")
    api_version: str = Field(..., description="API version")
    uptime_seconds: Optional[float] = Field(None, description="API uptime in seconds")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-07-13T08:32:03.456Z",
                "model_status": "healthy",
                "api_version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint"""

    model_type: str = Field(..., description="Type of ML model")
    model_version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Model training date")
    feature_count: int = Field(..., description="Number of features")
    classes: List[str] = Field(..., description="Model output classes")
    training_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    features: List[str] = Field(..., description="List of required features")

    class Config:
        schema_extra = {
            "example": {
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
                "features": ["avg_tx_amount", "device_change_freq", "..."]
            }
        }


class ErrorResponse(BaseModel):
    """Response model for API errors"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID if available")
    timestamp: datetime = Field(..., description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input: tx_hour must be between 0 and 23",
                "request_id": "req_1642086123456",
                "timestamp": "2025-07-13T08:32:03.456Z"
            }
        }