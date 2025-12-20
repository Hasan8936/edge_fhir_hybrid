from flask import Flask, request, jsonify
from app.edge_model import HybridDeployedModel
import numpy as np
import os

app = Flask(__name__)

# Initialize model
print("="*60)
print("🚀 INITIALIZING HYBRID DETECTION SYSTEM")
print("="*60)

try:
    model = HybridDeployedModel(models_dir="models")
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

# ======================== API ENDPOINTS ========================

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "service": "FHIR Hybrid Detection System",
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
        features = np.array(data["features"]).reshape(1, -1)
        metadata = data.get("metadata", {})
        
        # Run analysis
        result = model.analyze(features, meta=metadata)
        
        # Format response
        response = {
            "prediction": result['prediction'],
            "ae_score": result['ae_score'],
            "severity": result['severity'],
            "is_anomalous": result['is_anomalous'],
            "probabilities": result['probabilities'],
            "metadata": result['metadata']
        }
        
        # Log if anomalous
        if result['is_anomalous']:
            print(f"⚠️  ANOMALY DETECTED: {result['prediction']} (score: {result['ae_score']:.4f})")
        
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
        features = np.array([s["features"] for s in samples])
        
        # Batch prediction
        batch_results = model.predict_batch(features)
        
        # Format response
        results = []
        for i, sample in enumerate(samples):
            results.append({
                "prediction": batch_results['classes'][i],
                "ae_score": batch_results['ae_scores'][i],
                "probabilities": {
                    class_name: batch_results['probabilities'][i][j]
                    for j, class_name in enumerate(model.label_encoder.classes_)
                },
                "metadata": sample.get("metadata", {})
            })
        
        return jsonify({
            "count": len(results),
            "results": results
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """
    Get model information
    """
    return jsonify({
        "model": "RF + XGB + CNN AutoEncoder",
        "classes": list(model.label_encoder.classes_),
        "n_features": len(model.feature_mask),
        "rf_estimators": model.rf_model.n_estimators,
        "xgb_estimators": model.xgb_model.n_estimators
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