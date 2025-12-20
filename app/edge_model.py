import os
import numpy as np
import joblib
import pickle
from app.trt.ae_runtime import AERuntime


class HybridDeployedModel:
    """Hybrid inference model for Jetson Nano.

    - AutoEncoder (TensorRT) runs first for anomaly scoring
    - If AE exceeds low threshold, RF+XGB ensemble on CPU classifies
    """

    def __init__(self, models_dir="models"):
        print("[Hybrid Model] Loading artifacts...")
        self.models_dir = models_dir

        # Load scaler if present
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None

        # Feature mask (boolean or integer index list)
        self.feature_mask = np.load(os.path.join(models_dir, "feature_mask.npy"))

        # Label encoder
        with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        # RF + XGB (sklearn joblib)
        self.rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
        self.xgb_model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))

        # AutoEncoder TensorRT engine
        ae_engine = os.path.join(models_dir, "ae.engine")
        try:
            self.ae = AERuntime(ae_engine)
        except Exception as e:
            raise RuntimeError("Failed to initialize AE runtime: {}".format(e))

        print("[Hybrid Model] ✓ Loaded features: {}".format(self.feature_mask.shape))
        print("[Hybrid Model] ✓ Classes: {}".format(list(self.label_encoder.classes_)))

    def preprocess(self, X):
        """Scale and select features.

        Args:
            X: np.ndarray shape (n_samples, n_raw_features)

        Returns:
            np.ndarray shape (n_samples, n_selected_features)
        """
        X = np.array(X, dtype=np.float32)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Apply feature mask (boolean mask or index array)
        X_selected = X_scaled[:, self.feature_mask]
        return X_selected

    def infer(self, features, meta=None, thresholds=None):
        """Run the hybrid inference pipeline for a single sample.

        Args:
            features: list or np.ndarray of raw features (n_raw_features)
            meta: optional metadata dict
            thresholds: dict with keys 'low','medium','high'

        Returns:
            dict matching the required response format
        """
        if thresholds is None:
            thresholds = {"low": 0.01, "medium": 0.05, "high": 0.1}

        X = np.array(features).reshape(1, -1)
        X_sel = self.preprocess(X)

        # AutoEncoder score (reconstruction error)
        ae_score = float(self.ae.score(X_sel))
        print("[AE] score={:.6f} (selected_features={})".format(ae_score, X_sel.shape[1]))

        # Determine severity based on AE score
        if ae_score >= thresholds["high"]:
            sev = "HIGH"
        elif ae_score >= thresholds["medium"]:
            sev = "MEDIUM"
        else:
            sev = "LOW"

        all_results = {"autoencoder": {"ae_score": ae_score, "thresholds": thresholds}}

        # Fast-exit if AE indicates normal behaviour
        if ae_score < thresholds["low"]:
            print("[AE] below low threshold ({:.6f}) → fast-exit normal".format(thresholds["low"]))
            pred = "Normal"
            anom = False
            combined_score = ae_score
            all_results["rf_xgb"] = {
                "skipped": True
            }
        else:
            # Run RF and XGB on CPU
            rf_probs = self.rf_model.predict_proba(X_sel)[0].astype(float).tolist()
            xgb_probs = self.xgb_model.predict_proba(X_sel)[0].astype(float).tolist()
            # 50-50 ensemble
            ensemble = [(r + x) * 0.5 for r, x in zip(rf_probs, xgb_probs)]
            max_prob = max(ensemble)
            pred_idx = int(np.argmax(np.array(ensemble)))
            pred = str(self.label_encoder.inverse_transform([pred_idx])[0])
            print("[RF+XGB] pred={} max_prob={:.4f}".format(pred, max_prob))

            # Combine AE score and classifier confidence into unified anomaly score
            # (AE dominates; classifier adds weight based on 1 - confidence)
            combined_score = min(1.0, ae_score + (1.0 - max_prob) * 0.5)
            anom = (pred != "Normal") or (sev != "LOW")

            all_results["rf_xgb"] = {
                "pred": pred,
                "ensemble_probs": ensemble,
                "rf_probs": rf_probs,
                "xgb_probs": xgb_probs,
                "max_prob": float(max_prob)
            }

        # Final response format
        response = {
            "pred": pred,
            "score": float(combined_score),
            "sev": sev,
            "anom": bool(anom),
            "meta": meta or {},
            "all_results": all_results
        }

        return response
