from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .config import NORMAL_CLASS, SEV_HIGH, SEV_MED

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Wrap a deployed model to compute anomaly scores and severities.

    See module-level docstrings for details.
    """

    def __init__(self, model: Any, normal_class: str = NORMAL_CLASS, sev_high: float = SEV_HIGH, sev_med: float = SEV_MED) -> None:
        self.model = model
        self.normal_class = normal_class
        self.sev_high = float(sev_high)
        self.sev_med = float(sev_med)

    def _score(self, prob_vec: np.ndarray, classes: np.ndarray) -> float:
        try:
            cls_list = list(classes)
            if self.normal_class in cls_list:
                idx = cls_list.index(self.normal_class)
                return float(1.0 - prob_vec[idx])
        except Exception:
            logger.debug("Falling back to max-prob based score")
        return float(1.0 - float(np.max(prob_vec)))

    def _severity(self, score: float) -> str:
        if score >= self.sev_high:
            return "HIGH"
        if score >= self.sev_med:
            return "MEDIUM"
        return "LOW"

    def analyze(self, X: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
        labels, probs = self.model.predict(X)
        prob_vec = probs[0]
        classes = getattr(self.model.le, "classes_", None)
        if classes is None:
            logger.warning("Label encoder has no classes_; using ordinal indices")
            classes = np.arange(prob_vec.shape[0])

        score = self._score(prob_vec, classes)
        sev = self._severity(score)
        pred_label = labels[0]
        is_anom = (pred_label != self.normal_class) or (sev != "LOW")

        return {
            "pred": pred_label,
            "score": float(score),
            "sev": sev,
            "anom": bool(is_anom),
            "meta": meta,
        }
