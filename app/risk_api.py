#!/usr/bin/env python3
"""
=============================================================================
RISK API - BEHAVIORAL AUTHENTICATION WITH HTTPS & CORS
=============================================================================

Single-file Flask API for behavioral risk assessment with HTTPS, CORS and Swagger support
"""

from flask import Flask, request, jsonify, render_template_string
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

# OpenAPI/Swagger Documentation
SWAGGER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BMD Risk API - Swagger Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css" />
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin:0; background: #fafafa; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: '/swagger.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>
"""


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

            try:
                import joblib
                import pandas as pd
                import numpy as np

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

            feature_df = pd.DataFrame([features], columns=self.model_info['features'])
            prediction = self.model.predict(feature_df)[0]
            probabilities = self.model.predict_proba(feature_df)[0]

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

    def calculate_risk_score(self, transaction_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score from transaction features"""
        try:
            feature_mapping = {
                "avg_tx_amount": transaction_features.get("amount", 0.0),
                "device_change_freq": transaction_features.get("device_change_freq", 0.0),
                "tx_hour": transaction_features.get("transaction_hour", 12) / 23.0,
                "location_change_freq": transaction_features.get("location_change_freq", 0.0),
                "is_new_device": float(transaction_features.get("is_new_device", False)),
                "velocity_last_hour": transaction_features.get("velocity_last_hour", 0.0),
                "amount_deviation": transaction_features.get("amount_deviation", 0.0),
                "time_since_last_tx": transaction_features.get("time_since_last_tx", 0.0),
                "geo_velocity": transaction_features.get("geo_velocity", 0.0),
                "user_behavior_score": transaction_features.get("user_behavior_score", 0.5)
            }

            feature_vector = [
                feature_mapping["avg_tx_amount"],
                feature_mapping["device_change_freq"],
                feature_mapping["tx_hour"],
                feature_mapping["location_change_freq"],
                feature_mapping["is_new_device"],
                feature_mapping["velocity_last_hour"],
                feature_mapping["amount_deviation"],
                feature_mapping["time_since_last_tx"],
                feature_mapping["geo_velocity"],
                feature_mapping["user_behavior_score"]
            ]

            result = self.predict(feature_vector)

            risk_flag_mapping = {
                "legit": "legit",
                "suspicious": "suspicious",
                "fraud": "fraud"
            }

            risk_flag = risk_flag_mapping.get(result["prediction"], "suspicious")

            if result["prediction"] == "legit":
                risk_score = 1.0 - result["confidence"]
            elif result["prediction"] == "suspicious":
                risk_score = 0.5 + (result["confidence"] * 0.3)
            else:
                risk_score = 0.7 + (result["confidence"] * 0.3)

            return {
                "risk_score": float(risk_score),
                "risk_flag": risk_flag,
                "confidence": result["confidence"],
                "prediction_details": result,
                "feature_vector": feature_vector,
                "mock_mode": result.get("mock_mode", False)
            }

        except Exception as e:
            logger.error(f"Risk score calculation error: {e}")
            return {
                "risk_score": 1.0,
                "risk_flag": "fraud",
                "confidence": 0.0,
                "error": str(e),
                "mock_mode": True
            }


def init_model():
    global model_service
    model_service = ModelService()
    model_service.load_model()


# Swagger/OpenAPI JSON specification
@app.route('/swagger.json')
def swagger_json():
    """Return OpenAPI/Swagger JSON specification"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "BMD ML Risk API",
            "description": "Behavioral Authentication and Fraud Detection API using Machine Learning",
            "version": "1.2.0",
            "contact": {
                "name": "olafcio42",
                "url": "https://github.com/AI-Quantum-Tech-Security/BMD"
            }
        },
        "servers": [
            {
                "url": "https://localhost:5000",
                "description": "Development HTTPS server"
            },
            {
                "url": "http://localhost:5000",
                "description": "Development HTTP server"
            }
        ],
        "paths": {
            "/": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check API status and get service information",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "healthy"},
                                            "service": {"type": "string"},
                                            "version": {"type": "string"},
                                            "uptime_seconds": {"type": "number"},
                                            "timestamp": {"type": "string"},
                                            "model_loaded": {"type": "boolean"},
                                            "endpoints": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/risk-score": {
                "post": {
                    "summary": "Calculate Risk Score",
                    "description": "Analyze transaction features and return risk assessment with score and flag",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "amount": {
                                            "type": "number",
                                            "description": "Transaction amount",
                                            "example": 1500.50
                                        },
                                        "transaction_hour": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "maximum": 23,
                                            "description": "Hour of transaction (0-23)",
                                            "example": 14
                                        },
                                        "is_new_device": {
                                            "type": "boolean",
                                            "description": "Whether transaction is from a new device",
                                            "example": False
                                        },
                                        "device_change_freq": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Device change frequency (0.0-1.0)",
                                            "example": 0.2
                                        },
                                        "location_change_freq": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Location change frequency (0.0-1.0)",
                                            "example": 0.1
                                        },
                                        "velocity_last_hour": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Transaction velocity in last hour (0.0-1.0)",
                                            "example": 0.3
                                        },
                                        "amount_deviation": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Deviation from user's normal amounts (0.0-1.0)",
                                            "example": 0.4
                                        },
                                        "time_since_last_tx": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Time since last transaction (0.0-1.0)",
                                            "example": 0.5
                                        },
                                        "geo_velocity": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Geographic velocity (0.0-1.0)",
                                            "example": 0.05
                                        },
                                        "user_behavior_score": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                            "description": "Historical user behavior score (0.0-1.0)",
                                            "example": 0.8
                                        }
                                    },
                                    "required": ["amount", "transaction_hour"]
                                },
                                "examples": {
                                    "legitimate_transaction": {
                                        "summary": "Legitimate Transaction",
                                        "value": {
                                            "amount": 150.50,
                                            "transaction_hour": 14,
                                            "is_new_device": False,
                                            "device_change_freq": 0.1,
                                            "location_change_freq": 0.05,
                                            "velocity_last_hour": 0.2,
                                            "amount_deviation": 0.1,
                                            "time_since_last_tx": 0.3,
                                            "geo_velocity": 0.02,
                                            "user_behavior_score": 0.8
                                        }
                                    },
                                    "suspicious_transaction": {
                                        "summary": "Suspicious Transaction",
                                        "value": {
                                            "amount": 5000.00,
                                            "transaction_hour": 3,
                                            "is_new_device": True,
                                            "device_change_freq": 0.8,
                                            "location_change_freq": 0.9,
                                            "velocity_last_hour": 0.9,
                                            "amount_deviation": 0.9,
                                            "time_since_last_tx": 0.05,
                                            "geo_velocity": 0.8,
                                            "user_behavior_score": 0.1
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Risk assessment completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "risk_score": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                                "description": "Risk score (0.0 = lowest risk, 1.0 = highest risk)"
                                            },
                                            "risk_flag": {
                                                "type": "string",
                                                "enum": ["legit", "suspicious", "fraud"],
                                                "description": "Risk classification"
                                            },
                                            "confidence": {
                                                "type": "number",
                                                "description": "Model confidence in prediction"
                                            },
                                            "timestamp": {"type": "string"},
                                            "api_version": {"type": "string"},
                                            "mock_mode": {"type": "boolean"}
                                        }
                                    },
                                    "examples": {
                                        "legit_result": {
                                            "summary": "Legitimate Transaction Result",
                                            "value": {
                                                "risk_score": 0.23,
                                                "risk_flag": "legit",
                                                "confidence": 0.87,
                                                "timestamp": "2025-07-13T14:59:28.123456+00:00",
                                                "api_version": "1.2.0",
                                                "mock_mode": False
                                            }
                                        },
                                        "fraud_result": {
                                            "summary": "Fraudulent Transaction Result",
                                            "value": {
                                                "risk_score": 0.89,
                                                "risk_flag": "fraud",
                                                "confidence": 0.92,
                                                "timestamp": "2025-07-13T14:59:28.123456+00:00",
                                                "api_version": "1.2.0",
                                                "mock_mode": False
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid input data"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "summary": "Raw Feature Prediction (Legacy)",
                    "description": "Direct prediction using raw feature vector",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "features": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 10,
                                            "maxItems": 10,
                                            "description": "Array of 10 normalized features"
                                        }
                                    },
                                    "required": ["features"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prediction": {"type": "string"},
                                            "confidence": {"type": "number"},
                                            "risk_score": {"type": "number"},
                                            "probabilities": {"type": "object"},
                                            "mock_mode": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/model-info": {
                "get": {
                    "summary": "Model Information",
                    "description": "Get information about the loaded ML model",
                    "responses": {
                        "200": {
                            "description": "Model information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model_loaded": {"type": "boolean"},
                                            "model_type": {"type": "string"},
                                            "training_date": {"type": "string"},
                                            "feature_count": {"type": "integer"},
                                            "features": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "classes": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "training_metrics": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "TransactionFeatures": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "transaction_hour": {"type": "integer"},
                        "is_new_device": {"type": "boolean"},
                        "device_change_freq": {"type": "number"},
                        "location_change_freq": {"type": "number"},
                        "velocity_last_hour": {"type": "number"},
                        "amount_deviation": {"type": "number"},
                        "time_since_last_tx": {"type": "number"},
                        "geo_velocity": {"type": "number"},
                        "user_behavior_score": {"type": "number"}
                    }
                },
                "RiskAssessment": {
                    "type": "object",
                    "properties": {
                        "risk_score": {"type": "number"},
                        "risk_flag": {"type": "string"},
                        "confidence": {"type": "number"},
                        "timestamp": {"type": "string"},
                        "api_version": {"type": "string"}
                    }
                }
            }
        }
    })


@app.route('/docs')
def swagger_ui():
    """Serve Swagger UI documentation"""
    return render_template_string(SWAGGER_TEMPLATE)


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    return jsonify({
        "status": "healthy",
        "service": "ML Risk API - Behavioral Authentication",
        "version": "1.2.0",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_service.is_loaded if model_service else False,
        "endpoints": {
            "health": "/",
            "predict": "/predict",
            "risk_score": "/risk-score",
            "model_info": "/model-info",
            "docs": "/docs",
            "swagger": "/swagger.json"
        }
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Risk prediction endpoint (legacy)"""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        logger.info(f"Prediction request from {request.remote_addr}")
        data = request.get_json()
        logger.info(f"Request data: {data}")

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        features = data.get('features')
        if not features:
            return jsonify({"error": "No features provided"}), 400

        if not isinstance(features, list):
            return jsonify({"error": "Features must be a list"}), 400

        if len(features) != 10:
            return jsonify({"error": f"Expected 10 features, got {len(features)}"}), 400

        logger.info(f"Processing features: {features}")
        result = model_service.predict(features)
        logger.info(f"Prediction result: {result}")

        result.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "api_version": "1.2.0"
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


@app.route('/risk-score', methods=['POST', 'OPTIONS'])
def risk_score():
    """Risk score endpoint - accepts transaction features and returns risk assessment"""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        logger.info(f"Risk score request from {request.remote_addr}")
        data = request.get_json()
        logger.info(f"Transaction data: {data}")

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        if 'amount' not in data:
            return jsonify({"error": "Missing required field: amount"}), 400
        if 'transaction_hour' not in data:
            return jsonify({"error": "Missing required field: transaction_hour"}), 400

        result = model_service.calculate_risk_score(data)
        logger.info(f"Risk score result: {result}")

        response = {
            "risk_score": result["risk_score"],
            "risk_flag": result["risk_flag"],
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_version": "1.2.0",
            "mock_mode": result.get("mock_mode", False)
        }

        if "prediction_details" in result:
            response["prediction_details"] = result["prediction_details"]

        if "error" in result:
            response["error"] = result["error"]

        return jsonify(response)

    except Exception as e:
        logger.error(f"Risk score endpoint error: {e}")
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
            "api_version": "1.2.0"
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
        "available_endpoints": ["/", "/predict", "/risk-score", "/model-info", "/docs"]
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

    logger.warning("No SSL certificates found, generating self-signed certificate...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime as dt

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

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
    init_model()

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    ssl_context = create_ssl_context()

    if ssl_context:
        logger.info(f"Starting HTTPS server on {host}:{port}")
        logger.info("SSL/HTTPS enabled")
    else:
        logger.info(f"Starting HTTP server on {host}:{port}")
        logger.warning("SSL/HTTPS disabled - running in HTTP mode")

    logger.info("CORS enabled for all origins")
    logger.info("API Endpoints:")
    logger.info("Swagger UI: https://localhost:5000/docs")
    logger.info("Health check: GET /")
    logger.info("Risk score: POST /risk-score")
    logger.info("Prediction: POST /predict")
    logger.info("Model info: GET /model-info")

    app.run(
        host=host,
        port=port,
        debug=debug,
        ssl_context=ssl_context,
        threaded=True
    )