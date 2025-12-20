import os
import numpy as np
from app.trt.trt_runtime import TensorRTModel


class AERuntime:
    """AE runtime wrapper for TensorRT engine.

    Provides `score` and `score_batch` methods returning reconstruction MSE.
    """

    def __init__(self, engine_path):
        if not os.path.exists(engine_path):
            raise FileNotFoundError("TensorRT engine not found: {}".format(engine_path))

        # Load serialized engine via generic TRT runtime wrapper
        self.trt = TensorRTModel(engine_path)

        # Infer input/output shapes from engine bindings
        # binding order in TensorRTModel preserves input first
        self.input_shape = tuple(self.trt.inputs[0][0].shape) if hasattr(self.trt.inputs[0][0], 'shape') else None

    def _prepare_input(self, X):
        # Expect X shape: (n_selected_features,) or (1, n_selected_features)
        x = np.array(X, dtype=np.float32)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x.ravel()

        # Reshape to match exported AE input: (1,1,N,1)
        N = x.size
        inp = x.reshape(1, 1, N, 1).astype(np.float32)
        return inp

    def score(self, X):
        """Compute reconstruction MSE for a single sample.

        Args:
            X: Processed features (1, n_selected_features)

        Returns:
            float: mean squared error
        """
        inp = self._prepare_input(X)
        out = self.trt.predict(inp)

        # Ensure flattened arrays
        recon = np.array(out).ravel()
        orig = np.array(X).ravel()

        # Match length (sometimes TRT returns batch*elements)
        minlen = min(orig.size, recon.size)
        mse = float(np.mean((orig[:minlen] - recon[:minlen]) ** 2))
        return mse

    def score_batch(self, X_batch):
        """Compute MSE for a batch of samples.

        Args:
            X_batch: (n_samples, n_selected_features)

        Returns:
            np.ndarray: shape (n_samples,) of mse values
        """
        results = []
        for i in range(X_batch.shape[0]):
            results.append(self.score(X_batch[i]))
        return np.array(results)
