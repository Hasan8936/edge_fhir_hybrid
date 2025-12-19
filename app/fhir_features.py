import numpy as np
import hashlib
import joblib
import os

# Load scaler once to know expected feature length
_SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

try:
    _scaler = joblib.load(_SCALER_PATH)
    EXPECTED_FEATURES = int(_scaler.n_features_in_)
except Exception:
    # fallback (safe default)
    EXPECTED_FEATURES = 25


def hash_string(s, mod=10000):
    return int(hashlib.sha1(str(s).encode()).hexdigest(), 16) % mod


def extract_features(fhir: dict):
    """
    Convert FHIR AuditEvent JSON → fixed-length numeric vector
    """
    r = fhir or {}

    res = r.get("resourceType", "Unknown")
    act = r.get("action", "None")
    out = str(r.get("outcome", "0"))

    evt = r.get("event", {})
    tcode = evt.get("type", {}).get("code", "0")

    ag = (r.get("agent") or [{}])[0]
    user = ag.get("userId", "unknown")
    ip = ag.get("network", {}).get("address", "0.0.0.0")

    # -------- Core semantic features --------
    features = [
        hash_string(res),
        hash_string(act),
        hash_string(tcode),
        float(out) if out.replace(".", "", 1).isdigit() else 0.0,
        hash_string(user),
        hash_string(ip),
        float(len(r.get("agent", []))),
        float("fail" in str(r).lower()),
    ]

    feats = np.array(features, dtype=np.float32)

    # -------- PAD / TRUNCATE (CRITICAL FIX) --------
    if feats.shape[0] < EXPECTED_FEATURES:
        feats = np.pad(feats, (0, EXPECTED_FEATURES - feats.shape[0]))
    else:
        feats = feats[:EXPECTED_FEATURES]

    meta = {
        "resourceType": res,
        "action": act,
        "outcome": out,
        "user": user,
        "ip": ip,
        "feature_len": int(feats.shape[0]),
    }

    return feats, meta

