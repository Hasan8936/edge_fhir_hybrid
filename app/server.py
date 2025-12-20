from flask import Flask, request, jsonify
from app.edge_model import HybridDeployedModel
import numpy as np
import os
import datetime

app = Flask(__name__)

# Initialize model
print("=" * 60)
print("🚀 INITIALIZING HYBRID DETECTION SYSTEM")
print("=" * 60)

try:
    model = HybridDeployedModel()
    MODEL_READY = True
    print("✅ Model loaded successfully!\n")
except Exception as e:
    MODEL_READY = False
    model = None
    print("❌ Failed to load model: {}".format(e))
    # Keep server running but /health will report not ready

# ======================== API ENDPOINTS ========================

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy" if MODEL_READY else "degraded",
        "service": "FHIR Hybrid Detection System",
        "model_ready": bool(MODEL_READY),
        "version": "1.0.0"
    }), 200


@app.route("/fhir/notify", methods=["POST"])
def fhir_notify():
    """
    Main detection endpoint
    
    Expected JSON format:
    {
        "features": [f1, f2, ..., f_n],
        "metadata": {
            "patient_id": "...",
            "timestamp": "...",
            ...
        }
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or "features" not in data:
            return jsonify({
                "error": "Missing 'features' in request body"
            }), 400
        
        # Extract features
        features = data["features"]
        metadata = data.get("metadata", {})

        # Run hybrid inference
        result = model.infer(features, meta=metadata)

        # Persist alerts when anomalous
        if result.get("anom"):
            try:
                os.makedirs("logs", exist_ok=True)
                with open(os.path.join("logs", "alerts.log"), "a") as f:
                    f.write("{time} - ALERT - pred={pred} score={score:.6f} meta={meta}\n".format(
                        time=datetime.datetime.utcnow().isoformat(),
                        pred=result.get("pred"),
                        score=float(result.get("score", 0.0)),
                        meta=str(result.get("meta", {}))
                    ))
            except Exception:
                pass

        # Response must match required format
        response = {
            "pred": result.get("pred"),
            "score": float(result.get("score")),
            "sev": result.get("sev"),
            "anom": bool(result.get("anom")),
            "meta": result.get("meta"),
            "all_results": result.get("all_results")
        }

        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/fhir/batch", methods=["POST"])
def fhir_batch():
    """
    Batch detection endpoint
    
    Expected JSON format:
    {
        "samples": [
            {"features": [...], "metadata": {...}},
            {"features": [...], "metadata": {...}},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or "samples" not in data:
            return jsonify({
                "error": "Missing 'samples' in request body"
            }), 400
        
        samples = data["samples"]
        
        # Extract all features
        features = [s["features"] for s in samples]

        results = []
        for sample in samples:
            res = model.infer(sample.get("features"), meta=sample.get("metadata", {}))
            results.append(res)

        return jsonify({"count": len(results), "results": results}), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """
    Get model information
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model": "RF + XGB + CNN AutoEncoder",
        "classes": list(model.label_encoder.classes_),
        "n_features": len(model.feature_mask),
        "rf_estimators": getattr(model.rf, 'n_estimators', None),
        "xgb_estimators": getattr(model.xgb, 'n_estimators', None)
    }), 200


# ======================== RUN SERVER ========================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 Starting Flask server...")
    print("="*60)
    print("📍 Endpoints:")
    print("   - GET  /health         : Health check")
    print("   - POST /fhir/notify    : Single detection")
    print("   - POST /fhir/batch     : Batch detection")
    print("   - GET  /model/info     : Model information")
    print("="*60 + "\n")
    
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )