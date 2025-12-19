"""Train CNN Autoencoder on normal (non-anomalous) FHIR traffic.

The autoencoder learns to reconstruct normal network behavior.
During inference, high reconstruction error indicates anomaly.

This script is typically run ONCE during development.
For production on Jetson, pre-export ONNX model to models/cnn_ae.onnx
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Tuple, List
import os

logger = logging.getLogger(__name__)


class CNNAutoencoder(nn.Module):
    """Lightweight CNN Autoencoder optimized for Jetson Nano.
    
    Input: Feature vector of shape (batch, 1, feature_dim, 1)
    Output: Reconstructed features of same shape
    
    Architecture:
    - Encoder: 2 Conv layers + ReLU
    - Bottleneck: Fully connected layer
    - Decoder: 2 ConvTranspose layers
    
    Total params: ~15k (suitable for edge devices)
    """
    
    def __init__(self, input_dim: int = 25, latent_dim: int = 8):
        """
        Args:
            input_dim: Number of features from FHIR extraction (e.g., 25)
            latent_dim: Bottleneck dimension for compression
        """
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        
        # Bottleneck
        enc_out_size = (input_dim // 4) * 32
        self.fc_encode = nn.Linear(enc_out_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, enc_out_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 1), padding=(1, 0)),
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Sigmoid(),  # Normalize output to [0,1]
        )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        latent = self.fc_encode(encoded)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to feature space."""
        decoded = self.fc_decode(latent)
        decoded = decoded.view(decoded.size(0), 32, -1, 1)
        reconstructed = self.decoder(decoded)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode and decode."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


def train_autoencoder(
    normal_features: np.ndarray,
    input_dim: int = 25,
    latent_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> Tuple[CNNAutoencoder, List[float]]:
    """
    Train autoencoder on normal traffic features.
    
    Args:
        normal_features: (n_samples, input_dim) array of normal FHIR features
        input_dim: Feature vector size
        latent_dim: Bottleneck compression factor
        epochs: Training iterations
        batch_size: Batch size for SGD
        learning_rate: Adam learning rate
        device: "cpu" or "cuda"
        
    Returns:
        (trained_model, loss_history)
    """
    logger.info(f"Training CNN Autoencoder on {len(normal_features)} normal samples")
    logger.info(f"  Device: {device}, Latent dim: {latent_dim}, Epochs: {epochs}")
    
    # Create model
    model = CNNAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model = model.to(device)
    
    # Prepare data
    # Reshape (n, input_dim) → (n, 1, input_dim, 1) for Conv2d
    X_tensor = torch.FloatTensor(normal_features).unsqueeze(1).unsqueeze(-1)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (X_batch,) in enumerate(loader):
            X_batch = X_batch.to(device)
            
            # Forward pass
            X_recon = model(X_batch)
            loss = criterion(X_recon, X_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    logger.info(f"Training complete. Final loss: {loss_history[-1]:.6f}")
    return model, loss_history


def evaluate_threshold(
    model: CNNAutoencoder,
    normal_features: np.ndarray,
    anomaly_percentile: float = 95.0,
    device: str = "cpu",
) -> float:
    """
    Determine reconstruction error threshold for anomaly detection.
    
    Uses percentile of normal data reconstruction errors.
    
    Args:
        model: Trained autoencoder
        normal_features: Normal traffic features for threshold calibration
        anomaly_percentile: Percentile for threshold (e.g., 95 = top 5% normal data)
        device: Compute device
        
    Returns:
        Recommended MSE threshold
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(normal_features).unsqueeze(1).unsqueeze(-1)
        X_tensor = X_tensor.to(device)
        
        X_recon = model(X_tensor)
        
        # Compute per-sample reconstruction error
        mse_per_sample = torch.mean((X_tensor - X_recon) ** 2, dim=[1, 2, 3])
        mse_values = mse_per_sample.cpu().numpy()
    
    threshold = np.percentile(mse_values, anomaly_percentile)
    logger.info(f"Reconstruction error threshold (p{anomaly_percentile}): {threshold:.6f}")
    
    return threshold


if __name__ == "__main__":
    """Example: Train autoencoder on synthetic normal data."""
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic normal data (e.g., 1000 samples, 25 features)
    np.random.seed(42)
    normal_data = np.random.randn(1000, 25) * 0.1 + 0.5
    normal_data = np.clip(normal_data, 0, 1)  # Normalize to [0, 1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Train
    model, losses = train_autoencoder(
        normal_data,
        input_dim=25,
        latent_dim=8,
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        device=device,
    )
    
    # Evaluate threshold
    threshold = evaluate_threshold(model, normal_data, anomaly_percentile=95, device=device)
    
    logger.info(f"Model ready. Save with: torch.save(model.state_dict(), 'models/cnn_ae.pth')")
