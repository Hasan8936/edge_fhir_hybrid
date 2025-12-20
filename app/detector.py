import numpy as np


class EdgeDetector:
    """Detector orchestrates the hybrid pipeline using AE-first logic.

    If AE error < low threshold -> fast-exit as Normal. Otherwise runs RF+XGB.
    """

    def __init__(self, model, thresholds=None):
        self.model = model
        if thresholds is None:
            thresholds = {"low": 0.01, "medium": 0.05, "high": 0.1}
        self.thresholds = thresholds

    def analyze(self, X, meta=None):
        # X expected shape (1, n_features) or list
        result = self.model.infer(X[0] if isinstance(X, (list, tuple, np.ndarray)) and np.array(X).ndim == 2 else X, meta=meta, thresholds=self.thresholds)
        return {
            "pred": result.get("pred"),
            "score": result.get("score"),
            "sev": result.get("sev"),
            "anom": result.get("anom"),
            "meta": result.get("meta"),
            "all_results": result.get("all_results")
        }


