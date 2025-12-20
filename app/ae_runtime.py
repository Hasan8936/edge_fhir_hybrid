import torch
import torch.nn as nn
import numpy as np

FEATURES = 25
LATENT_DIM = 8

class CNNAutoEncoder(nn.Module):
    """
    AutoEncoder architecture (must match training)
    """
    def __init__(self, input_dim, latent_dim):
        super(CNNAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class AERuntime:
    """
    AutoEncoder runtime for anomaly scoring
    """
    def __init__(self, model_path):
        """
        Load trained AutoEncoder model
        
        Args:
            model_path: Path to ae.pth file
        """
        self.model = CNNAutoEncoder(input_dim=FEATURES, latent_dim=LATENT_DIM)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"[AE Runtime] Model loaded on {self.device}")
    
    def score(self, X):
        """
        Compute reconstruction error (anomaly score)
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        
        Returns:
            float: Mean squared reconstruction error
        """
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Reconstruct
            reconstructed = self.model(X_tensor)
            
            # Compute MSE
            mse = torch.mean((X_tensor - reconstructed) ** 2).item()
        
        return mse
    
    def score_batch(self, X):
        """
        Compute per-sample reconstruction errors
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        
        Returns:
            numpy array of shape (n_samples,): Per-sample MSE scores
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            
            # Per-sample MSE
            mse_per_sample = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        return mse_per_sample