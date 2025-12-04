from flask import Flask,request,jsonify
import os,json
from datetime import datetime
import numpy as np
from .edge_model import HybridDeployedModel
from .detector import EdgeDetector
from .fhir_features import extract_features
from .config import LOG_FILE
app=Flask(__name__)
model=HybridDeployedModel()
det=EdgeDetector(model)
os.makedirs(os.path.dirname(LOG_FILE),exist_ok=True)
def log_alert(e): open(LOG_FILE,"a").write(json.dumps(e)+"\n")
@app.route("/health") 
def h(): return {"status":"ok"},200
@app.route("/fhir/notify",methods=["POST"])
"""Flask HTTP API for receiving FHIR notifications and running inference.

Endpoints
---------
- `GET /health` simple health check.
- `POST /fhir/notify` accepts a FHIR resource (JSON) and returns prediction.
- `GET /config` returns non-sensitive runtime configuration (model classes,
  normal class name).

Security notes
--------------
This service should be run behind an API gateway or reverse proxy that
terminates TLS and enforces authentication. Avoid sending raw FHIR bodies
to logs or stdout to reduce PHI/PII leakage.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify
import numpy as np

from .config import LOG_FILE, ensure_paths_exist, MODELS_DIR, NORMAL_CLASS
from .edge_model import HybridDeployedModel, ModelLoadError
from .detector import EdgeDetector
from .fhir_features import extract_features

# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _append_alert(record: Dict[str, Any]) -> None:
    """Append a JSON-line record to the alerts log.

    This intentionally only writes minimal fields and should not contain full
    FHIR payloads.
    """
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("Failed to write alert to log file %s", LOG_FILE)


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Application factory for the edge inference service.

    Parameters
    ----------
    test_config: Optional[dict]
        If provided, may override configuration for tests.
    """
    ensure_paths_exist()

    app = Flask(__name__)

    # Try to instantiate model; if model artifacts are missing the server
    # still starts but `/fhir/notify` will return 503 until artifacts are
    # available. This avoids container restart loops while operators fix
    # file mounts.
    model = None
    detector = None
    try:
        model = HybridDeployedModel()
        detector = EdgeDetector(model)
        logger.info("Model loaded successfully from %s", MODELS_DIR)
    except ModelLoadError as exc:
        logger.error("Model artifacts not available: %s", exc)


    @app.route("/health", methods=["GET"])
    def health() -> Any:  # simple health check
        return jsonify({"status": "ok"}), 200


    @app.route("/config", methods=["GET"])
    def config_info() -> Any:
        classes = None
        if model is not None:
            try:
                classes = list(getattr(model.le, "classes_", []))
            except Exception:
                classes = None
        return jsonify({"models_dir": MODELS_DIR, "normal_class": NORMAL_CLASS, "classes": classes}), 200


    @app.route("/fhir/notify", methods=["POST"])
    def fhir_notify() -> Any:
        # Basic content validation
        if not request.content_type or "application/json" not in request.content_type:
            return jsonify({"error": "Content-Type must be application/json"}), 415

        if detector is None:
            return jsonify({"error": "Model artifacts not loaded"}), 503

        try:
            data = request.get_json(force=True)
        except Exception:
            logger.exception("Malformed JSON in request")
            return jsonify({"error": "Malformed JSON"}), 400

        # Extract features and metadata (this function is robust to missing fields)
        try:
            feats, meta = extract_features(data)
        except Exception:
            logger.exception("Feature extraction failed")
            return jsonify({"error": "Feature extraction failed"}), 400

        # Attach remote address if available (best-effort)
        try:
            meta["remote_addr"] = request.remote_addr
        except Exception:
            meta["remote_addr"] = None

        # Ensure correct shape
        X = np.asarray(feats).reshape(1, -1)
        try:
            res = detector.analyze(X, meta)
        except Exception:
            logger.exception("Model inference failed")
            return jsonify({"error": "Model inference failed"}), 500

        if res.get("anom"):
            alert = {"ts": datetime.utcnow().isoformat() + "Z", "pred": res.get("pred"), "score": res.get("score"), "sev": res.get("sev"), "meta": res.get("meta")}
            _append_alert(alert)
            logger.warning("Anomaly detected: %s", {"pred": alert["pred"], "score": alert["score"], "sev": alert["sev"]})

        return jsonify(res), 200

    # Placeholder for metrics endpoint or Prometheus client integration. Add
    # instrumentation here (counters for requests, anomalies, latencies, etc.)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001)
