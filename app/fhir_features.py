import numpy as np, hashlib
def hash_string(s, mod=10000):
    return int(hashlib.sha1(str(s).encode()).hexdigest(),16)%mod
def extract_features(fhir):
    r=fhir or {}
    res=r.get("resourceType","Unknown")
    act=r.get("action","None")
    out=str(r.get("outcome","0"))
    evt=r.get("event",{})
    tcode=evt.get("type",{}).get("code","0")
    ag=r.get("agent",[{}])[0]
    user=ag.get("userId","unknown")
    ip=ag.get("network",{}).get("address","0.0.0.0")
    feats=np.array([
        hash_string(res),
        hash_string(act),
        hash_string(tcode),
        float(out) if out.replace('.','',1).isdigit() else 0.0,
        hash_string(user),
        hash_string(ip),
        float(len(r.get("agent",[]))),
        float("fail" in str(r).lower())
    ],dtype=np.float32)
    meta={"resourceType":res,"action":act,"outcome":out,"user":user,"ip":ip}
    return feats,meta
"""FHIR feature extraction helpers.

This module provides a small, extensible feature extraction function that
turns incoming FHIR JSON (AuditEvent and similar) into a numeric vector that
the on-device models can consume.

Notes
-----
- Only a small, non-exhaustive set of fields is mapped to numeric features.
- Avoid logging full FHIR content to prevent PHI/PII leakage. Only minimal
  metadata (hashed) is logged when needed.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Tuple, Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Define which features we currently extract. This makes it easier to extend
# later and to keep consistent ordering for the model.
FEATURE_NAMES: List[str] = [
    "resourceType_hash",
    "action_hash",
    "event_type_code_hash",
    "outcome_value",
    "user_hash",
    "ip_hash",
    "agent_count",
    "failure_flag",
]


def hash_string(value: Any, mod: int = 100000) -> int:
    """Deterministically hash a value to an integer in [0, mod).

    Parameters
    ----------
    value: Any
        Value to hash. Converted to str before hashing.
    mod: int
        Modulus for the output to keep values bounded.

    Returns
    -------
    int
        Hash value in range [0, mod).
    """
    s = str(value)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def extract_features(fhir_json: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract numeric features and metadata from a FHIR resource dict.

    The function is intentionally tolerant to missing fields and unknown
    resource shapes. It returns a numeric vector (numpy array) and a small
    metadata dictionary with non-sensitive identifying fields.

    Parameters
    ----------
    fhir_json: Dict[str, Any]
        Parsed JSON body of the incoming FHIR resource (usually AuditEvent).

    Returns
    -------
    features: np.ndarray
        1D float32 array with the features in the order defined by
        `FEATURE_NAMES`.
    meta: Dict[str, Any]
        Minimal metadata useful for logging and debugging (resourceType,
        action, user (hashed), ip (masked/hashes)).
    """
    if not isinstance(fhir_json, dict):
        logger.warning("Expected FHIR JSON object, got %s", type(fhir_json))
        fhir_json = {}

    # Top-level fields
    resource_type = fhir_json.get("resourceType", "Unknown")
    action = fhir_json.get("action", "")

    # outcome might be numeric or string; try to convert
    outcome = fhir_json.get("outcome")
    outcome_val = _safe_float(outcome, default=0.0)

    # event block (AuditEvent structure)
    event_block = fhir_json.get("event") or {}
    event_type_code = ""
    if isinstance(event_block, dict):
        event_type = event_block.get("type") or {}
        if isinstance(event_type, dict):
            event_type_code = event_type.get("code", "")

    # agent array
    agents = fhir_json.get("agent") or []
    agent_count = float(len(agents))
    user_val = ""
    ip_val = ""
    if isinstance(agents, list) and len(agents) > 0:
        first_agent = agents[0] or {}
        user_val = first_agent.get("userId", "")
        network = first_agent.get("network") or {}
        if isinstance(network, dict):
            ip_val = network.get("address", "")

    # simple failure detection flag: look for keywords in the resource text
    text_str = ""
    if isinstance(fhir_json, dict):
        # avoid logging this; only use to detect 'fail' indicators
        try:
            text_str = str(fhir_json)
        except Exception:
            text_str = ""
    failure_flag = 1.0 if "fail" in text_str.lower() or "denied" in text_str.lower() else 0.0

    # build numeric vector in the canonical order
    features = np.array([
        float(hash_string(resource_type)),
        float(hash_string(action)),
        float(hash_string(event_type_code)),
        outcome_val,
        float(hash_string(user_val)),
        float(hash_string(ip_val)),
        agent_count,
        failure_flag,
    ], dtype=np.float32)

    # metadata only with minimal non-PII info; user/ip are hashed
    meta = {
        "resourceType": resource_type,
        "action": action,
        "outcome": outcome_val,
        "user_hash": hash_string(user_val),
        "ip_hash": hash_string(ip_val),
        "agent_count": int(agent_count),
    }

    return features, meta
