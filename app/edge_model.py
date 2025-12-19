# app/edge_model.py

import joblib
import numpy as np
import pickle
import os

from app.trt.trt_runtime import TensorRTXGB

MODELS_DIR = "models"

class HybridDeployedModel:
    def __init__(self):
        self.rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        self.mask = np.load(os.path.join(MODELS_DIR, "feature_mask.npy"))

        with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)

        # TensorRT XGBoost
        self.trt_xgb = TensorRTXGB(
            os.path.join(MODELS_DIR, "xgb_model.engine")
        )

        # ensemble weights
        self.RW = 0.5
        self.XW = 0.5

    def predict_proba(self, X):
        # scale + select features
        Xs = self.scaler.transform(X)
        Xs = Xs[:, self.mask]

        # RF (CPU)
        pr = self.rf.predict_proba(Xs)

        # XGB (TensorRT GPU)
        px = self.trt_xgb.predict_proba(Xs).reshape(1, -1)

        probs = self.RW * pr + self.XW * px
        return probs, self.le.classes_
