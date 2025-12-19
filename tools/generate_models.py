"""Generate dummy model artifacts for testing the edge inference service."""
import os
import sys
import pickle
import numpy as np

# Create models directory if needed
os.makedirs('models', exist_ok=True)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib
    
    print("Generating dummy model artifacts...")
    
    # Create dummy training data
    X_train = np.random.randn(100, 8)
    y_train = np.array(['Normal'] * 70 + ['Attack'] * 30)
    
    # 1. RandomForest
    print("  - Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/rf_model.pkl')
    print("    ✓ Saved models/rf_model.pkl")
    
    # 2. XGBoost (if available)
    try:
        import xgboost as xgb
        print("  - Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, 
                                      use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, 'models/xgb_model.pkl')
        print("    ✓ Saved models/xgb_model.pkl")
    except ImportError:
        print("    ! XGBoost not available, creating dummy pickle instead")
        # Create a dummy object that quacks like XGBoost
        class DummyXGB:
            def predict_proba(self, X):
                return np.random.dirichlet([1, 1], X.shape[0])
        joblib.dump(DummyXGB(), 'models/xgb_model.pkl')
        print("    ✓ Saved models/xgb_model.pkl (dummy)")
    
    # 3. StandardScaler
    print("  - Creating StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("    ✓ Saved models/scaler.pkl")
    
    # 4. Feature mask
    print("  - Creating feature mask...")
    feature_mask = np.array([True, True, True, True, True, True, True, True], dtype=bool)
    np.save('models/feature_mask.npy', feature_mask)
    print("    ✓ Saved models/feature_mask.npy")
    
    # 5. LabelEncoder
    print("  - Creating LabelEncoder...")
    le = LabelEncoder()
    le.fit(y_train)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("    ✓ Saved models/label_encoder.pkl")
    
    print("\n✅ All model artifacts created!")
    print(f"   Classes: {list(le.classes_)}")
    print(f"   Feature mask: {feature_mask.shape[0]} features")
    
except ImportError as e:
    print(f"❌ Error: Required package not installed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
