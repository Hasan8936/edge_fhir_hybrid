"""
Minimal dummy model loader for testing without actual sklearn/xgboost.
This creates simple pickle-compatible objects that satisfy the interface.
"""
import pickle
import numpy as np
import os

os.makedirs('models', exist_ok=True)

# Create a minimal RandomForest-like object
class DummyRF:
    def __init__(self):
        self.n_estimators = 10
    
    def predict_proba(self, X):
        """Return dummy probabilities: 70% Normal, 30% Attack"""
        n_samples = X.shape[0]
        normal_prob = np.random.uniform(0.6, 0.9, n_samples)
        attack_prob = 1.0 - normal_prob
        return np.column_stack([normal_prob, attack_prob])

# Create a minimal XGBoost-like object
class DummyXGB:
    def __init__(self):
        self.n_estimators = 10
    
    def predict_proba(self, X):
        """Return dummy probabilities: slightly different distribution"""
        n_samples = X.shape[0]
        normal_prob = np.random.uniform(0.65, 0.85, n_samples)
        attack_prob = 1.0 - normal_prob
        return np.column_stack([normal_prob, attack_prob])

# Create a minimal StandardScaler-like object
class DummyScaler:
    def __init__(self):
        self.mean_ = np.zeros(8)
        self.scale_ = np.ones(8)
    
    def transform(self, X):
        """Simple z-score scaling"""
        return (X - self.mean_) / (self.scale_ + 1e-8)

# Create a minimal LabelEncoder-like object
class DummyLabelEncoder:
    def __init__(self):
        self.classes_ = np.array(['Normal', 'Attack'])
    
    def inverse_transform(self, y):
        """Map indices back to labels"""
        return np.array([self.classes_[i] for i in y])

print("Creating dummy model artifacts...")

# Save RF model
rf = DummyRF()
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("✓ models/rf_model.pkl")

# Save XGB model
xgb = DummyXGB()
with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
print("✓ models/xgb_model.pkl")

# Save scaler
scaler = DummyScaler()
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ models/scaler.pkl")

# Save feature mask
feature_mask = np.array([True, True, True, True, True, True, True, True], dtype=bool)
np.save('models/feature_mask.npy', feature_mask)
print("✓ models/feature_mask.npy")

# Save label encoder
le = DummyLabelEncoder()
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ models/label_encoder.pkl")

print("\n✅ All dummy model artifacts created successfully!")
print("   Classes: ['Normal', 'Attack']")
print("   Features: 8")
