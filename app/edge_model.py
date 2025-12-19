"""
Hybrid classification model combining RandomForest and XGBoost.

PRODUCTION-GRADE EDGE ARCHITECTURE:
- RandomForest (RF): CPU-based, low latency, reliable
- XGBoost (XGB): CPU-based, high accuracy
- CNN Autoencoder: TensorRT-accelerated (separate module in app/cnn/)

RF+XGB OUTPUT: Classification (Normal, DDoS, etc.)
CNN AE OUTPUT: Reconstruction error (anomaly likelihood)

This module handles ONLY the tree ensemble classification.
CNN inference is managed separately in app/cnn/trt_runtime.py
to avoid complex TensorRT conversions of tree models.

Security note: Never attempt to convert tree models to TensorRT.
They are best deployed on CPU using native inference engines.
"""

import joblib
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = "models"


class HybridDeployedModel:
    """
    CPU-based RandomForest + XGBoost ensemble.
    
    Features:
    - Loads pre-trained models from .pkl files
    - Applies feature scaling
    - Selects GOA-identified features via mask
    - Ensemble prediction via weighted average
    - No GPU/TensorRT involved (trees don't benefit)
    
    Attributes:
        rf: Trained RandomForest classifier
        xgb: Trained XGBoost classifier
        scaler: Feature StandardScaler
        mask: Boolean array selecting important features
        le: Label encoder (maps class indices to names)
        rf_weight: Weight for RF in ensemble [0, 1]
        xgb_weight: Weight for XGB in ensemble [0, 1]
    """
    
    def __init__(self):
        """
        Initialize hybrid model from saved artifacts.
        
        Raises:
            FileNotFoundError: If any required model file missing
            ValueError: If models cannot be loaded
        """
        logger.info("Initializing HybridDeployedModel...")
        
        # Load components
        try:
            self.rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
            logger.info("✓ RandomForest model loaded")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"RandomForest model not found: {MODELS_DIR}/rf_model.pkl"
            )
        
        try:
            self.xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
            logger.info("✓ XGBoost model loaded")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"XGBoost model not found: {MODELS_DIR}/xgb_model.pkl"
            )
        
        try:
            self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
            logger.info("✓ Feature scaler loaded")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler not found: {MODELS_DIR}/scaler.pkl"
            )
        
        try:
            self.mask = np.load(os.path.join(MODELS_DIR, "feature_mask.npy"))
            logger.info(f"✓ Feature mask loaded ({np.sum(self.mask)} features selected)")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Feature mask not found: {MODELS_DIR}/feature_mask.npy"
            )
        
        try:
            with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
                self.le = pickle.load(f)
            logger.info(f"✓ Label encoder loaded: {list(self.le.classes_)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Label encoder not found: {MODELS_DIR}/label_encoder.pkl"
            )
        
        # Ensemble weights
        # Tuned for this specific dataset (can be optimized via validation)
        self.rf_weight = 0.5
        self.xgb_weight = 0.5
        
        logger.info(f"HybridDeployedModel initialized successfully")
        logger.info(f"  Ensemble weights: RF={self.rf_weight}, XGB={self.xgb_weight}")
    
    def predict_proba(self, X: np.ndarray) -> tuple:
        """
        Predict class probabilities using weighted ensemble.
        
        Args:
            X: Input features of shape (n_samples, n_features_all)
               Will be scaled and feature-selected
        
        Returns:
            (probabilities, class_names) where:
            - probabilities: (n_samples, n_classes) probability matrix
            - class_names: List of class names
        """
        # Feature preprocessing
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.mask]
        
        # RF prediction
        rf_probs = self.rf.predict_proba(X_selected)
        
        # XGB prediction
        xgb_probs = self.xgb.predict_proba(X_selected)
        
        # Ensemble: weighted average
        ensemble_probs = (
            self.rf_weight * rf_probs + 
            self.xgb_weight * xgb_probs
        )
        
        return ensemble_probs, self.le.classes_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (argmax of probabilities).
        
        Args:
            X: Input features of shape (n_samples, n_features_all)
        
        Returns:
            Predicted class labels
        """
        probs, _ = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """
        Predict classes with confidence scores.
        
        Args:
            X: Input features
        
        Returns:
            (predicted_labels, confidences, class_names) where:
            - predicted_labels: Class indices
            - confidences: Max probability per sample
            - class_names: List of all class names
        """
        probs, class_names = self.predict_proba(X)
        predicted_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        return predicted_indices, confidences, class_names
