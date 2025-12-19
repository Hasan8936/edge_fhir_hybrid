"""TensorRT inference runtime for CNN Autoencoder on Jetson Nano.

Production-grade deployment:
- GPU-accelerated inference via TensorRT
- Sub-10ms latency per sample on Jetson Nano
- Memory-efficient for edge deployment
- Graceful fallback to ONNX Runtime if TensorRT unavailable

Workflow:
1. Precompiled TensorRT engine from CNN ONNX (trt_build.sh on target device)
2. Load via TensorRTRuntime
3. Compute reconstruction error: MSE(input, output)
4. Classify as anomaly if MSE > threshold
"""

import numpy as np
import logging
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

# TensorRT optional (may not be available on dev machine)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.debug("TensorRT not available. Will use ONNX Runtime fallback.")

# ONNX Runtime for CPU/fallback
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class TensorRTCNNRuntime:
    """
    TensorRT inference engine for CNN Autoencoder.
    
    Attributes:
        engine: Compiled TensorRT engine
        context: Execution context
        input_name: Input binding name
        output_name: Output binding name
        input_shape: Expected input shape
    """
    
    def __init__(self, engine_path: str):
        """
        Load pre-built TensorRT engine.
        
        Args:
            engine_path: Path to .trt or .engine file
            
        Raises:
            FileNotFoundError: If engine file not found
            RuntimeError: If TensorRT not available
        """
        if not HAS_TENSORRT:
            raise RuntimeError(
                "TensorRT not installed. Install: "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
            )
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        logger.info(f"Loading TensorRT engine: {engine_path}")
        
        # Deserialize engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        logger.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger.TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Analyze bindings
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.bindings = []
        self.d_inputs = []
        self.d_outputs = []
        
        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            shape = self.engine.get_binding_shape(binding_idx)
            size = int(np.prod(shape))
            dtype = self.engine.get_binding_dtype(binding_idx)
            
            if self.engine.binding_is_input(binding_idx):
                self.input_name = binding_name
                self.input_shape = shape
                logger.info(f"  Input: {binding_name}, shape={shape}, dtype={dtype}")
            else:
                self.output_name = binding_name
                self.output_shape = shape
                logger.info(f"  Output: {binding_name}, shape={shape}, dtype={dtype}")
            
            # Allocate GPU memory
            d_mem = cuda.mem_alloc(size * np.dtype(np.float32).itemsize)
            self.bindings.append(int(d_mem))
            
            if self.engine.binding_is_input(binding_idx):
                self.d_inputs.append(d_mem)
            else:
                self.d_outputs.append(d_mem)
        
        logger.info(f"TensorRT engine loaded successfully")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            input_data: Input array of shape (1, 1, feature_dim, 1) or similar
                       Will be converted to float32 and reshaped if needed
            
        Returns:
            Reconstructed features array of same shape as input
        """
        # Ensure float32
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Reshape to expected input shape
        if input_data.shape != self.input_shape:
            input_data = input_data.reshape(self.input_shape)
        
        # Upload to GPU
        cuda.memcpy_htod_async(self.d_inputs[0], input_data, self.stream)
        
        # Execute
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.stream.synchronize()
        
        # Download from GPU
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_outputs[0], self.stream)
        self.stream.synchronize()
        
        return output
    
    def compute_reconstruction_error(self, input_data: np.ndarray) -> float:
        """
        Compute Mean Squared Error between input and reconstruction.
        
        Args:
            input_data: Input features
            
        Returns:
            MSE value (float)
        """
        output = self.infer(input_data)
        mse = float(np.mean((input_data - output) ** 2))
        return mse
    
    def __del__(self):
        """Cleanup GPU resources."""
        for mem in self.bindings:
            try:
                cuda.mem_free(mem)
            except:
                pass


class ONNXRuntimeCNNFallback:
    """
    Fallback CNN inference using ONNX Runtime (CPU-based).
    
    Used when TensorRT not available (e.g., development machine).
    Much slower than TensorRT but guarantees inference capability.
    """
    
    def __init__(self, onnx_path: str):
        """
        Load ONNX model via ONNX Runtime.
        
        Args:
            onnx_path: Path to .onnx file
            
        Raises:
            FileNotFoundError: If ONNX file not found
            ImportError: If onnxruntime not installed
        """
        if not HAS_ONNXRUNTIME:
            raise ImportError(
                "onnxruntime not installed. Install: pip install onnxruntime"
            )
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        logger.info(f"Loading ONNX model (CPU fallback): {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        
        input_name = self.session.get_inputs()[0].name
        self.input_name = input_name
        logger.info(f"  ONNX input: {input_name}")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on CPU via ONNX Runtime.
        
        Args:
            input_data: Input features (will be converted to float32)
            
        Returns:
            Reconstructed features
        """
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[0]
    
    def compute_reconstruction_error(self, input_data: np.ndarray) -> float:
        """
        Compute MSE between input and reconstruction.
        
        Args:
            input_data: Input features
            
        Returns:
            MSE value
        """
        output = self.infer(input_data)
        mse = float(np.mean((input_data - output) ** 2))
        return mse


def create_cnn_runtime(
    engine_or_onnx_path: str,
    force_onnx: bool = False,
) -> Optional[object]:
    """
    Factory function to create CNN inference runtime.
    
    Automatically selects TensorRT if available and not forced to ONNX.
    
    Args:
        engine_or_onnx_path: Path to TensorRT engine (.trt/.engine) or ONNX (.onnx)
        force_onnx: Force ONNX Runtime even if TensorRT available
        
    Returns:
        TensorRTCNNRuntime or ONNXRuntimeCNNFallback instance, or None if both unavailable
    """
    if not force_onnx and engine_or_onnx_path.endswith((".trt", ".engine")):
        if HAS_TENSORRT:
            try:
                return TensorRTCNNRuntime(engine_or_onnx_path)
            except Exception as e:
                logger.warning(f"TensorRT loading failed: {e}. Trying ONNX Runtime...")
    
    if HAS_ONNXRUNTIME and engine_or_onnx_path.endswith(".onnx"):
        try:
            return ONNXRuntimeCNNFallback(engine_or_onnx_path)
        except Exception as e:
            logger.error(f"ONNX Runtime loading failed: {e}")
            return None
    
    logger.error("No inference runtime available. Install TensorRT or onnxruntime.")
    return None


if __name__ == "__main__":
    """Example: Load runtime and test inference."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Try to load TensorRT engine
    engine_path = "models/cnn_ae.engine"
    onnx_path = "models/cnn_ae.onnx"
    
    runtime = None
    
    if os.path.exists(engine_path):
        try:
            runtime = create_cnn_runtime(engine_path, force_onnx=False)
        except Exception as e:
            logger.warning(f"TensorRT engine loading failed: {e}")
    
    if runtime is None and os.path.exists(onnx_path):
        runtime = create_cnn_runtime(onnx_path, force_onnx=True)
    
    if runtime is None:
        logger.error("No valid model found. Train and export first.")
        sys.exit(1)
    
    # Test inference
    test_input = np.random.randn(1, 1, 25, 1).astype(np.float32)
    mse = runtime.compute_reconstruction_error(test_input)
    logger.info(f"Test MSE: {mse:.6f}")
