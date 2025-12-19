from flask import Flask, request, jsonify
from datetime import datetime
import os, json, sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    __package__ = "app"

from .edge_model import HybridDeployedModel
from .detector import EdgeDetector
from .fhir_features import extract_features

LOG_FILE = "alerts.log"

app = Flask(__name__)

model = HybridDeployedModel()
detector = EdgeDetector(model)

@app.route("/health")
def health():
    return {"status": "ok", "model_ready": True}, 200


@app.route("/fhir/notify", methods=["POST"])
def notify():
    if not request.is_json:
        return jsonify({"error": "JSON required"}), 415

    fhir = request.get_json()
    feats, meta = extract_features(fhir)

    X = feats.reshape(1, -1)
    raw = detector.analyze(X, meta)

    # 🔴 NORMALIZE RESPONSE (THIS IS THE FIX)
    response = {
        "pred": raw.get("pred"),
        "score": raw.get("score"),
        "sev": raw.get("sev"),
        "anom": raw.get("anom", False),
        "meta": raw.get("meta", {}),
        "all_results": raw.get("all_results", {})
    }

    # log only anomalies
    if response["anom"]:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "ts": datetime.utcnow().isoformat() + "Z",
                **response
            }) + "\n")

    return jsonify(response), 200



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)


