import numpy as np

class EdgeDetector:
    def __init__(self, model, normal="Normal", hi=0.95, mid=0.85):
        self.model = model
        self.normal = normal
        self.hi = hi
        self.mid = mid

    def _score(self, probs, classes):
        if self.normal in classes:
            idx = list(classes).index(self.normal)
            return 1.0 - float(probs[idx])
        return 1.0 - float(np.max(probs))

    def _severity(self, score):
        if score >= self.hi:
            return "HIGH"
        if score >= self.mid:
            return "MEDIUM"
        return "LOW"

    def analyze(self, X, meta=None):
        probs, classes = self.model.predict_proba(X)
        probs = probs[0]

        pred = classes[np.argmax(probs)]
        score = self._score(probs, classes)
        sev = self._severity(score)
        anom = (pred != self.normal) or (sev != "LOW")

        return {
            "pred": str(pred),
            "score": float(score),
            "sev": sev,
            "anom": bool(anom),
            "meta": meta or {},
            "all_results": {
                "default": {
                    "classes": list(classes),
                    "probs": probs.tolist()
                }
            }
        }


