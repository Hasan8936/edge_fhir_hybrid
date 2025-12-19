"""Export trained CNN Autoencoder to ONNX format.

ONNX enables:
1. Cross-platform deployment
2. TensorRT conversion for GPU acceleration on Jetson
3. Framework-agnostic inference

Typical workflow:
1. Train on development machine (train_autoencoder.py)
2. Export to ONNX (this script)
3. Build TensorRT engine on target Jetson device
4. Deploy trt_runtime.py for production inference
"""

import torch
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    input_dim: int = 25,
    onnx_path: str = "models/cnn_ae.onnx",
    opset_version: int = 11,
) -> str:
    """
    Export CNN Autoencoder to ONNX format.
    
    Args:
        model: Trained CNNAutoencoder instance
        input_dim: Feature dimension (must match training)
        onnx_path: Output ONNX file path
        opset_version: ONNX opset (11 = good TensorRT support)
        
    Returns:
        Path to exported ONNX file
    """
    logger.info(f"Exporting model to ONNX: {onnx_path}")
    
    model.eval()
    
    # Create dummy input matching (batch_size=1, 1, input_dim, 1)
    dummy_input = torch.randn(1, 1, input_dim, 1)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["reconstructed"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "reconstructed": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )
    
    logger.info(f"ONNX export complete: {onnx_path}")
    return onnx_path


def verify_onnx(onnx_path: str, input_dim: int = 25) -> bool:
    """
    Verify ONNX model integrity and test inference.
    
    Args:
        onnx_path: Path to ONNX file
        input_dim: Expected input dimension
        
    Returns:
        True if ONNX model is valid and runnable
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnx or onnxruntime not installed. Skipping verification.")
        return True
    
    # Check ONNX model syntax
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    logger.info("✓ ONNX model syntax valid")
    
    # Test inference
    session = ort.InferenceSession(onnx_path)
    
    # Dummy input (batch_size=1, 1, input_dim, 1)
    test_input = np.random.randn(1, 1, input_dim, 1).astype(np.float32)
    outputs = session.run(None, {"features": test_input})
    
    logger.info(f"✓ ONNX inference test passed. Output shape: {outputs[0].shape}")
    return True


if __name__ == "__main__":
    """Example: Load trained model and export to ONNX."""
    import sys
    from train_autoencoder import CNNAutoencoder
    
    logging.basicConfig(level=logging.INFO)
    
    # Path to trained model weights
    model_path = "models/cnn_ae.pth"
    onnx_path = "models/cnn_ae.onnx"
    
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found: {model_path}")
        logger.info("Run train_autoencoder.py first to train the model.")
        sys.exit(1)
    
    # Load trained model
    logger.info(f"Loading model from {model_path}")
    model = CNNAutoencoder(input_dim=25, latent_dim=8)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Export to ONNX
    export_to_onnx(model, input_dim=25, onnx_path=onnx_path)
    
    # Verify ONNX
    verify_onnx(onnx_path, input_dim=25)
    
    logger.info(f"✓ Model ready for TensorRT conversion: {onnx_path}")
