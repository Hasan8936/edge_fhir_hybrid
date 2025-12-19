"""
Production-grade FHIR security monitoring service with hybrid ML.

Endpoint: POST /fhir/notify
- Input: FHIR AuditEvent JSON
- Feature extraction: 25 GOA-selected features
- Classification: RandomForest + XGBoost ensemble (CPU)
- Anomaly detection: CNN Autoencoder reconstruction error (TensorRT)
- Output: Combined prediction with severity levels

Alert thresholds:
- LOW:    Known benign class OR MSE < 0.05
- MEDIUM: Suspicious class OR 0.05 <= MSE < 0.15
- HIGH:   Known attack class OR MSE >= 0.15

Security: All requests logged to alerts.log for SOC integration
"""

from flask import Flask, request, jsonify
import os
import json
from datetime import datetime
import numpy as np
import sys
import logging

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    __package__ = "app"

from .edge_model import HybridDeployedModel
from .fhir_features import extract_features
from .config import LOG_FILE
from .cnn.trt_runtime import create_cnn_runtime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============= CONFIGURATION =============
USE_MOCK_MODEL = False  # 🔴 Set to True for testing without ML models
CNN_MODEL_PATH = "models/cnn_ae.onnx"  # ONNX or TensorRT engine
MSE_THRESHOLD_LOW = 0.05
MSE_THRESHOLD_HIGH = 0.15
# =========================================

# Global model instances
hybrid_model = None
cnn_runtime = None
CNN_READY = False
MODEL_READY = False

def initialize_models():
    """Load ML models at startup."""
    global hybrid_model, cnn_runtime, CNN_READY, MODEL_READY
    
    logger.info("Initializing ML models...")
    
    # Load RF + XGB
    try:
        hybrid_model = HybridDeployedModel()
        MODEL_READY = True
        logger.info("✓ Hybrid classifier initialized")
    except Exception as e:
        logger.error(f"✗ Failed to load hybrid model: {e}")
        MODEL_READY = False
    
    # Load CNN (optional but recommended)
    if os.path.exists(CNN_MODEL_PATH):
        try:
            cnn_runtime = create_cnn_runtime(CNN_MODEL_PATH, force_onnx=False)
            CNN_READY = True
            logger.info(f"✓ CNN Autoencoder loaded: {CNN_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"✗ CNN Autoencoder not available: {e}")
            logger.warning("  Proceeding with RF/XGB only. Anomaly scores will be classification-based.")
            CNN_READY = False
    else:
        logger.warning(f"CNN model not found: {CNN_MODEL_PATH}")
        logger.warning("  Proceeding with RF/XGB only.")
        CNN_READY = False

# Initialize on startup
initialize_models()

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_alert(alert_record: dict) -> None:
    """
    Log alert to JSONL file for Grafana/Loki ingestion.
    
    Args:
        alert_record: Dictionary with alert metadata
    """
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert_record, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to log alert: {e}")


def compute_severity(pred_class: str, mse: float, confidence: float) -> str:
    """
    Determine alert severity based on ML outputs.
    
    Severity logic:
    - HIGH:   Known attack class (e.g., "DDoS") OR high reconstruction error
    - MEDIUM: Suspicious class (e.g., "Suspicious") OR borderline MSE
    - LOW:    Normal or low reconstruction error
    
    Args:
        pred_class: Predicted class name from RF/XGB
        mse: Reconstruction error from CNN (0.0 if CNN unavailable)
        confidence: Confidence score from RF/XGB
    
    Returns:
        Severity level: "LOW", "MEDIUM", or "HIGH"
    """
    # Known attack classes
    attack_classes = {"DDoS", "ScanPort", "Infiltration", "Malware"}
    
    # Check if predicted class is known attack
    if pred_class in attack_classes and confidence > 0.7:
        return "HIGH"
    
    # Check CNN reconstruction error (if available)
    if CNN_READY and mse > MSE_THRESHOLD_HIGH:
        return "HIGH"
    
    if CNN_READY and mse > MSE_THRESHOLD_LOW:
        return "MEDIUM"
    
    # Suspicious classes
    suspicious_classes = {"Suspicious", "Anomaly", "Unknown"}
    if pred_class in suspicious_classes:
        return "MEDIUM"
    
    return "LOW"


def reshape_for_cnn(features: np.ndarray) -> np.ndarray:
    """
    Reshape feature vector for CNN input.
    
    CNN expects: (batch_size, channels=1, height=feature_dim, width=1)
    
    Args:
        features: (feature_dim,) or (1, feature_dim) array
    
    Returns:
        (1, 1, feature_dim, 1) array for CNN
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim == 2:
        features = features.reshape(1, 1, features.shape[-1], 1)
    return features


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "classifier_ready": MODEL_READY,
        "cnn_ready": CNN_READY,
        "mock_mode": USE_MOCK_MODEL,
    }), 200


@app.route("/fhir/notify", methods=["POST"])
def fhir_notify():
    """
    Process FHIR AuditEvent and return security classification.
    
    Request:
        POST /fhir/notify
        Content-Type: application/json
        Body: FHIR AuditEvent JSON
    
    Response:
        {
            "pred": "<class>",
            "score": <float>,
            "sev": "<LOW|MEDIUM|HIGH>",
            "anom": <bool>,
            "meta": {
                "feature_len": 25,
                "ip": "192.168.1.10",
                ...
            },
            "classifier": {
                "rf": <prob>,
                "xgb": <prob>,
                "ensemble": <prob>
            },
            "cnn": {
                "mse": <float>,
                "available": <bool>
            },
            "all_results": {...}
        }
    """
    try:
        # Validate content type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        
        data = request.get_json()
        logger.debug(f"Received FHIR event: {data.get('action', '?')}")
        
        # Mock mode (for testing without trained models)
        if USE_MOCK_MODEL:
            return _mock_response(data)
        
        # Feature extraction
        try:
            features, meta = extract_features(data)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return jsonify({"error": "Feature extraction failed", "details": str(e)}), 400
        
        # Ensure features are float32 and reshaped correctly
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # ===== CLASSIFIER: RF + XGB Ensemble =====
        classifier_results = {}
        if MODEL_READY:
            try:
                pred_indices, confidences, class_names = hybrid_model.predict_with_confidence(features)
                pred_idx = int(pred_indices[0])
                pred_class = str(class_names[pred_idx])
                confidence = float(confidences[0])
                
                classifier_results = {
                    "pred_class": pred_class,
                    "pred_index": pred_idx,
                    "confidence": confidence,
                    "class_names": list(class_names),
                }
                logger.info(f"Classification: {pred_class} ({confidence:.3f})")
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                pred_class = "Unknown"
                confidence = 0.0
        else:
            pred_class = "Unknown"
            confidence = 0.0
            logger.warning("Classifier not ready; using fallback")
        
        # ===== ANOMALY DETECTION: CNN Autoencoder =====
        mse = 0.0
        cnn_available = False
        if CNN_READY:
            try:
                features_for_cnn = reshape_for_cnn(features)
                mse = cnn_runtime.compute_reconstruction_error(features_for_cnn)
                cnn_available = True
                logger.info(f"CNN MSE: {mse:.6f}")
            except Exception as e:
                logger.warning(f"CNN inference failed: {e}")
                mse = 0.0
                cnn_available = False
        
        # ===== ANOMALY DECISION =====
        is_anomaly = (
            (pred_class in {"DDoS", "ScanPort", "Infiltration", "Malware"}) or
            (cnn_available and mse > MSE_THRESHOLD_HIGH)
        )
        
        # ===== SEVERITY =====
        severity = compute_severity(pred_class, mse, confidence)
        
        # ===== BUILD RESPONSE =====
        response = {
            "pred": pred_class,
            "score": float(confidence),
            "sev": severity,
            "anom": is_anomaly,
            "meta": meta,
            "classifier": classifier_results,
            "cnn": {
                "mse": float(mse),
                "available": cnn_available,
                "threshold_low": MSE_THRESHOLD_LOW,
                "threshold_high": MSE_THRESHOLD_HIGH,
            },
            "all_results": {
                "features_extracted": int(features.shape[-1]),
            }
        }
        
        # ===== LOGGING =====
        if is_anomaly:
            alert_record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "pred": pred_class,
                "sev": severity,
                "confidence": confidence,
                "mse": float(mse),
                "meta": meta,
            }
            log_alert(alert_record)
            logger.warning(f"ALERT LOGGED: {severity} severity, {pred_class}")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.exception(f"Unexpected error in /fhir/notify: {e}")
        return jsonify({
            "error": "internal_error",
            "message": str(e),
        }), 500


def _mock_response(fhir_data: dict) -> tuple:
    """Generate mock response for testing."""
    try:
        features, meta = extract_features(fhir_data)
    except:
        meta = {}
    
    response = {
        "pred": "DDoS",
        "score": 0.93,
        "sev": "HIGH",
        "anom": True,
        "meta": meta,
        "classifier": {"pred_class": "DDoS", "confidence": 0.93},
        "cnn": {"mse": 0.25, "available": False},
        "all_results": {},
    }
    return jsonify(response), 200


if __name__ == "__main__":
    logger.info("Starting Edge FHIR Security Server on 0.0.0.0:5001")
    logger.info(f"  Mock mode: {USE_MOCK_MODEL}")
    logger.info(f"  Classifier ready: {MODEL_READY}")
    logger.info(f"  CNN ready: {CNN_READY}")
    app.run(host="0.0.0.0", port=5001, debug=False)


