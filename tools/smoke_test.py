#!/usr/bin/env python3
"""Smoke test to verify model artifacts can be loaded and run inference."""
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_model_loading():
    """Test that all model artifacts can be loaded."""
    print("=" * 60)
    print("SMOKE TEST: Model Loading and Inference")
    print("=" * 60)
    
    try:
        from app.edge_model import HybridDeployedModel, ModelLoadError
        from app.detector import EdgeDetector
        from app.fhir_features import extract_features
        import numpy as np
        
        print("\n[1/3] Testing model loading...")
        try:
            model = HybridDeployedModel()
            print("    ✓ Model loaded successfully")
        except ModelLoadError as e:
            print(f"    ✗ Model loading failed: {e}")
            print("    ! This is expected if model artifacts are not yet present.")
            print("    ! Run: python generate_dummy_models.py")
            return False
        
        print("\n[2/3] Testing feature extraction...")
        test_fhir = {
            "resourceType": "AuditEvent",
            "action": "E",
            "outcome": 0,
            "agent": [{"userId": "test_user", "network": {"address": "192.168.1.1"}}],
            "event": {"type": {"code": "login"}}
        }
        features, meta = extract_features(test_fhir)
        print(f"    ✓ Extracted {len(features)} features")
        print(f"    ✓ Metadata: {meta}")
        
        print("\n[3/3] Testing inference...")
        detector = EdgeDetector(model)
        X = features.reshape(1, -1)
        result = detector.analyze(X, meta)
        print(f"    ✓ Prediction: {result['pred']}")
        print(f"    ✓ Score: {result['score']:.3f}")
        print(f"    ✓ Severity: {result['sev']}")
        print(f"    ✓ Anomaly: {result['anom']}")
        
        print("\n" + "=" * 60)
        print("✅ SMOKE TEST PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
