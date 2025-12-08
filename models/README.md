# Model Artifacts

This directory contains trained model artifacts that are loaded at runtime by the edge inference service.

## Required Files

The service expects the following files to be present:

1. **rf_model.pkl** - Trained RandomForest classifier (scikit-learn)
2. **xgb_model.pkl** - Trained XGBoost classifier
3. **scaler.pkl** - Feature scaler (StandardScaler or similar)
4. **feature_mask.npy** - Boolean NumPy array indicating which features to use
5. **label_encoder.pkl** - LabelEncoder for class labels (scikit-learn)

## Generating Dummy Models for Testing

If you want to test the service without a production model, run the generation script:

```bash
python ../generate_dummy_models.py
```

This will create minimal dummy models that satisfy the interface but return random predictions.

## Training Your Own Models

For production use, train models offline using your training pipeline:

```python
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

# Assume X_train, y_train are prepared
X_train = ...  # shape: (n_samples, n_features)
y_train = ...  # class labels: array of strings or ints

# 1. Train RandomForest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, 'models/rf_model.pkl')

# 2. Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                               use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'models/xgb_model.pkl')

# 3. Fit and save scaler
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'models/scaler.pkl')

# 4. Create and save feature mask (e.g., after feature selection)
selected_features = np.array([True, True, True, True, True, True, True, True])
np.save('models/feature_mask.npy', selected_features)

# 5. Fit and save label encoder
le = LabelEncoder()
le.fit(y_train)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
```

## Format Requirements

- **rf_model.pkl**: Must have `predict_proba(X)` method returning (n_samples, n_classes)
- **xgb_model.pkl**: Must have `predict_proba(X)` method returning (n_samples, n_classes)
- **scaler.pkl**: Must have `transform(X)` method
- **feature_mask.npy**: Boolean array of shape (n_features,). True = keep, False = drop
- **label_encoder.pkl**: Must have `classes_` attribute and `inverse_transform(y)` method

## Deployment Notes

When deploying to Jetson:

1. Train and export models on your development machine (CPU or GPU with enough memory)
2. Copy the 5 files into the `models/` directory
3. Ensure proper file permissions (read access for the container)
4. Mount the `models/` directory as read-only: `./models:/opt/app/models:ro`

The service will load these artifacts at startup and use them for all predictions.
