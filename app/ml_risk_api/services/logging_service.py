"""
Logging service for ML Risk API
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import os


def setup_logging():
    """Setup structured logging for the API"""

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Setup file handler
    file_handler = logging.FileHandler("logs/ml_risk_api.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    api_logger = logging.getLogger("ml_risk_api")
    api_logger.setLevel(logging.INFO)


def log_request(
        request_id: str,
        endpoint: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None
):
    """
    Log API request details for debugging and monitoring

    Args:
        request_id: Unique request identifier
        endpoint: API endpoint called
        input_data: Request input data
        output_data: Response output data (if available)
    """
    logger = logging.getLogger("ml_risk_api.requests")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "endpoint": endpoint,
        "input_features": {
            "feature_count": len(input_data),
            "sample_features": {
                k: v for k, v in list(input_data.items())[:5]  # Log first 5 features
            }
        }
    }

    if output_data:
        if "error" in output_data:
            log_entry["status"] = "error"
            log_entry["error"] = output_data["error"]
        else:
            log_entry["status"] = "success"
            log_entry["output"] = {
                "risk_score": output_data.get("risk_score"),
                "risk_flag": output_data.get("risk_flag"),
                "confidence": output_data.get("confidence"),
                "processing_time_ms": output_data.get("processing_time_ms")
            }

    logger.info(json.dumps(log_entry))


class APIMetrics:
    """Simple metrics collector for API monitoring"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.risk_flag_counts = {"legit": 0, "suspicious": 0, "fraud": 0}

    def record_request(self, processing_time: float, risk_flag: str = None, error: bool = False):
        """Record request metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time

        if error:
            self.error_count += 1
        elif risk_flag in self.risk_flag_counts:
            self.risk_flag_counts[risk_flag] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0 else 0
        )

        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_processing_time_ms": avg_processing_time,
            "risk_flag_distribution": self.risk_flag_counts,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global metrics instance
api_metrics = APIMetrics()