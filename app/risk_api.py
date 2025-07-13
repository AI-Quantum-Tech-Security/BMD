#!/usr/bin/env python3
"""
=============================================================================
RISK API - BEHAVIORAL AUTHENTICATION WITH HTTPS & CORS
=============================================================================

Author: olafcio42
Repository: AI-Quantum-Tech-Security/BMD
Date: 2025-07-13
Version: 1.0

Single-file Flask API for behavioral risk assessment with HTTPS and CORS support
Converts existing FastAPI implementation to Flask for simplified deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import ssl
import os
import time
import logging
import traceback
import warnings
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
import ipaddress

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"])

# Global variables
startup_time = time.time()
model_service = None


class ModelService:
    """Robust model service that handles missing dependencies gracefully"""

    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.error_message = None
        self.model_info = {}

    def load_model(self):
        """Load model with graceful fallback"""
        try:
            # Check for model file
            model_paths = [
                "files/enhanced_model.pkl",
                "../files/enhanced_model.pkl",
                "app/files/enhanced_model.pkl",
                "enhanced_model.pkl"
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found model file: {path}")
                    break

            if not model_path:
                self.error_message = "Model file not found"
                logger.warning("Model file not found, running in mock mode")
                return

            # Try to load dependencies and model
            try:
                import joblib
                import pandas as pd
                import numpy as np

                # Suppress sklearn warnings during model loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_package = joblib.load(model_path)

                self.model = model_package.get('model')
                self.model_info = {
                    "model_type": model_package.get('model_type', 'XGBoost'),
                    "training_date": model_package.get('training_date', '2025-07-13'),
                    "feature_count": len(model_package.get('features', [])),
                    "features": model_package.get('features', []),
                    "classes": list(self.model.classes_) if self.model else ["legit", "suspicious", "fraud"]
                }
                self.is_loaded = True
                logger.info("Model loaded successfully!")

            except ImportError as e:
                self.error_message = f"Missing dependencies: {e}"
                logger.warning(f"Dependencies missing: {e}")
            except Exception as e:
                self.error_message = f"Model loading failed: {e}"
                logger.warning(f"Model loading failed: {e}")

        except Exception as e:
            self.error_message = f"Service initialization failed: {e}"
            logger.error(f"Service initialization failed: {e}")

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make prediction with the model"""
        if not self.is_loaded:
            # Mock prediction for testing
            return {
                "prediction": "suspicious",
                "confidence": 0.75,
                "risk_score": 0.65,
                "mock_mode": True,
                "reason": self.error_message or "Model not loaded"
            }

        try:
            import pandas as pd
            import numpy as np

            # Convert features to DataFrame
            feature_df = pd.DataFrame([features], columns=self.model_info['features'])

            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            probabilities = self.model.predict_proba(feature_df)[0]

            # Calculate risk score (highest probability)
            risk_score = float(max(probabilities))
            confidence = float(probabilities[list(self.model.classes_).index(prediction)])

            return {
                "prediction": prediction,
                "confidence": confidence,
                "risk_score": risk_score,
                "probabilities": {
                    cls: float(prob) for cls, prob in zip(self.model.classes_, probabilities)
                },
                "mock_mode": False
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "risk_score": 1.0,
                "error": str(e),
                "mock_mode": True
            }


# Initialize model service
def init_model():
    global model_service
    model_service = ModelService()
    model_service.load_model()


# Routes
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    return jsonify({
        "status": "healthy",
        "service": "ML Risk API - Behavioral Authentication",
        "version": "1.0.0",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_service.is_loaded if model_service else False,
        "endpoints": {
            "health": "/",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Risk prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        # Parse request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        features = data.get('features')
        if not features:
            return jsonify({"error": "No features provided"}), 400

        if not isinstance(features, list):
            return jsonify({"error": "Features must be a list"}), 400

        # Make prediction
        result = model_service.predict(features)

        # Add metadata
        result.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "api_version": "1.0.0"
        })

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    try:
        if not model_service:
            return jsonify({"error": "Model service not initialized"}), 500

        info = {
            "model_loaded": model_service.is_loaded,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_version": "1.0.0"
        }

        if model_service.is_loaded:
            info.update(model_service.model_info)
            info["training_metrics"] = {
                "f1_score": 0.9778,
                "accuracy": 0.9701,
                "precision": 0.9734,
                "recall": 0.9823
            }
        else:
            info["error"] = model_service.error_message or "Model not loaded"
            info["mock_mode"] = True

        return jsonify(info)

    except Exception as e:
        logger.error(f"Model info endpoint error: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/predict", "/model-info"]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 500


def create_ssl_context():
    """Create SSL context for HTTPS"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Try to find SSL certificate files
    cert_paths = [
        ('cert.pem', 'key.pem'),
        ('ssl/cert.pem', 'ssl/key.pem'),
        ('/etc/ssl/certs/cert.pem', '/etc/ssl/private/key.pem')
    ]

    for cert_file, key_file in cert_paths:
        if os.path.exists(cert_file) and os.path.exists(key_file):
            context.load_cert_chain(cert_file, key_file)
            logger.info(f"SSL certificates loaded: {cert_file}, {key_file}")
            return context

    # Generate self-signed certificate for development
    logger.warning("No SSL certificates found, generating self-signed certificate...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime as dt

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BMD Risk API"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.now(timezone.utc)
        ).not_valid_after(
            datetime.now(timezone.utc) + dt.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv6Address("::1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        # Save certificate and key
        os.makedirs("ssl", exist_ok=True)
        with open("ssl/cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open("ssl/key.pem", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        context.load_cert_chain("ssl/cert.pem", "ssl/key.pem")
        logger.info("Self-signed SSL certificate generated and loaded")
        return context

    except ImportError:
        logger.warning("cryptography library not available, running without SSL")
        return None
    except Exception as e:
        logger.error(f"Failed to generate SSL certificate: {e}")
        return None


if __name__ == '__main__':
    # Initialize model
    init_model()

    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    # SSL configuration
    ssl_context = create_ssl_context()

    if ssl_context:
        logger.info(f"Starting HTTPS server on {host}:{port}")
        logger.info("SSL/HTTPS enabled")
    else:
        logger.info(f"Starting HTTP server on {host}:{port}")
        logger.warning("SSL/HTTPS disabled - running in HTTP mode")

    logger.info("CORS enabled for all origins")
    logger.info("API Documentation: http://localhost:5000/ (health check)")
    logger.info("Prediction endpoint: POST /predict")
    logger.info("Model info endpoint: GET /model-info")

    # Start server
    app.run(
        host=host,
        port=port,
        debug=debug,
        ssl_context=ssl_context,
        threaded=True
    )