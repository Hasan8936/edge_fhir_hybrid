"""Model loader and inference helper for the deployed hybrid model.

This module is responsible for loading pre-trained artifacts from disk and
exposing a small API for inference. It is intentionally lightweight and does
not attempt any training.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pickle

from .config import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a required model artifact cannot be loaded."""


class HybridDeployedModel:
    """Load and serve a hybrid ensemble of RandomForest and XGBoost.

    Parameters
    ----------
    rf_weight: float
        Weight for the RandomForest predictions (XGBoost gets 1 - rf_weight).
    models_dir: Optional[str]
        Directory where model artifacts live. Defaults to `MODELS_DIR`.
    """

    def __init__(self, rf_weight: float = 0.5, models_dir: Optional[str] = None) -> None:
        self.rf_weight = float(rf_weight)
        self.xgb_weight = 1.0 - self.rf_weight
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.rf: Any = None
        self.xgb: Any = None
        self.scaler: Any = None
        self.mask: Optional[np.ndarray] = None
        self.le: Any = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model artifacts from disk and validate their presence.

        Raises
        ------
        ModelLoadError
            If any required file is missing or cannot be loaded.
        """
        try:
            rf_path = self.models_dir / "rf_model.pkl"
            xgb_path = self.models_dir / "xgb_model.pkl"
            scaler_path = self.models_dir / "scaler.pkl"
            mask_path = self.models_dir / "feature_mask.npy"
            le_path = self.models_dir / "label_encoder.pkl"

            for p in (rf_path, xgb_path, scaler_path, mask_path, le_path):
                if not p.exists():
                    raise ModelLoadError(f"Missing model artifact: {p}")

            logger.info("Loading RandomForest model from %s", rf_path)
            self.rf = joblib.load(str(rf_path))

            logger.info("Loading XGBoost model from %s", xgb_path)
            self.xgb = joblib.load(str(xgb_path))

            logger.info("Loading scaler from %s", scaler_path)
            self.scaler = joblib.load(str(scaler_path))

            logger.info("Loading feature mask from %s", mask_path)
            self.mask = np.load(str(mask_path)).astype(bool)

            logger.info("Loading label encoder from %s", le_path)
            with open(str(le_path), "rb") as f:
                self.le = pickle.load(f)
        except Exception as exc:  # keep broad to wrap unknown errors
            logger.exception("Failed loading model artifacts: %s", exc)
            raise ModelLoadError(str(exc)) from exc

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Scale and apply the feature mask to an input array.

        Parameters
        ----------
        X: np.ndarray
            2D array of raw features with shape (n_samples, n_raw_features).

        Returns
        -------
        np.ndarray
            Transformed feature array suitable for prediction.
        """
        if self.scaler is None or self.mask is None:
            raise ModelLoadError("Model scaler or mask not loaded")
        Xs = self.scaler.transform(X)
        # mask must index the second dimension
        try:
            return Xs[:, self.mask]
        except Exception as exc:
            logger.exception("Feature mask application failed: %s", exc)
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble predicted probabilities for input X.

        Parameters
        ----------
        X: np.ndarray
            2D array (n_samples, n_raw_features)

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with probabilities.
        """
        Xs = self.preprocess(X)
        pr = self.rf.predict_proba(Xs)
        px = self.xgb.predict_proba(Xs)
        # ensure shapes match
        if pr.shape != px.shape:
            raise RuntimeError("RF and XGB predict_proba returned different shapes")
        return self.rf_weight * pr + self.xgb_weight * px

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return predicted labels and probability vectors.

        Returns
        -------
        labels: np.ndarray
            Decoded labels (strings) for each sample.
        probs: np.ndarray
            Probability vectors for each sample.
        """
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        labels = self.le.inverse_transform(idx)
        return labels, probs

