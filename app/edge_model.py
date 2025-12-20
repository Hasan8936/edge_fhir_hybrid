import numpy as np
import joblib
import pickle
import os
from app.ae_runtime import AERuntime

class HybridDeployedModel:
    """
    Complete hybrid detection system:
    - RF + XGB for attack classification
    - AutoEncoder for anomaly scoring
    """
    
    def __init__(self, models_dir="models"):
        """
        Load all trained models and preprocessors
        """
        print("[Hybrid Model] Loading artifacts...")
        
        # Load preprocessors
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.feature_mask = np.load(os.path.join(models_dir, "feature_mask.npy"))
        
        # Load label encoder
        with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)
        
        # Load RF and XGB models
        self.rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
        self.xgb_model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))
        
        # Load AutoEncoder
        self.ae = AERuntime(os.path.join(models_dir, "ae.pth"))
        
        print(f"[Hybrid Model] ✓ Loaded {len(self.feature_mask)} selected features")
        print(f"[Hybrid Model] ✓ Classes: {list(self.label_encoder.classes_)}")
    
    def preprocess(self, X):
        """
        Apply scaling and feature selection
        
        Args:
            X: Raw features (n_samples, all_features)
        
        Returns:
            Processed features (n_samples, selected_features)
        """
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Select features
        X_selected = X_scaled[:, self.feature_mask]
        
        return X_selected
    
    def predict(self, X):
        """
        Predict attack class and compute anomaly score
        
        Args:
            X: Raw features (n_samples, all_features)
        
        Returns:
            tuple: (predicted_class, anomaly_score, attack_probabilities)
        """
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Get predictions from RF and XGB
        rf_probs = self.rf_model.predict_proba(X_processed)
        xgb_probs = self.xgb_model.predict_proba(X_processed)
        
        # Ensemble (50-50 blend)
        ensemble_probs = 0.5 * rf_probs + 0.5 * xgb_probs
        
        # Get predicted class
        predicted_class_idx = np.argmax(ensemble_probs, axis=1)
        predicted_class = self.label_encoder.inverse_transform(predicted_class_idx)
        
        # Get anomaly score from AutoEncoder
        ae_score = self.ae.score(X_processed)
        
        # Return results
        return (
            predicted_class[0],
            float(ae_score),
            ensemble_probs[0].tolist()
        )
    
    def predict_batch(self, X):
        """
        Predict for multiple samples
        
        Args:
            X: Raw features (n_samples, all_features)
        
        Returns:
            dict: Batch prediction results
        """
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Get predictions
        rf_probs = self.rf_model.predict_proba(X_processed)
        xgb_probs = self.xgb_model.predict_proba(X_processed)
        ensemble_probs = 0.5 * rf_probs + 0.5 * xgb_probs
        
        # Classes
        predicted_classes = self.label_encoder.inverse_transform(
            np.argmax(ensemble_probs, axis=1)
        )
        
        # Anomaly scores
        ae_scores = self.ae.score_batch(X_processed)
        
        return {
            'classes': predicted_classes.tolist(),
            'ae_scores': ae_scores.tolist(),
            'probabilities': ensemble_probs.tolist()
        }
    
    def analyze(self, X, meta=None, thresholds={'low': 0.01, 'medium': 0.05, 'high': 0.1}):
        """
        Complete analysis with severity assessment
        
        Args:
            X: Raw features (1, all_features)
            meta: Optional metadata dict
            thresholds: AE score thresholds
        
        Returns:
            dict: Complete analysis results
        """
        pred_class, ae_score, probs = self.predict(X)
        
        # Determine severity based on AE score
        if ae_score >= thresholds['high']:
            severity = "HIGH"
        elif ae_score >= thresholds['medium']:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Check if anomalous
        is_anomalous = (pred_class != "Normal") or (severity != "LOW")
        
        return {
            'prediction': pred_class,
            'ae_score': ae_score,
            'severity': severity,
            'is_anomalous': is_anomalous,
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.label_encoder.classes_, probs)
            },
            'metadata': meta or {}
        }