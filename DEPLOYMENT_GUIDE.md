"""
Production Deployment Guide: Edge FHIR Hybrid on Jetson Nano

This guide covers:
1. Model Training & Export
2. TensorRT Engine Building
3. Docker Build & Deployment
4. Monitoring Setup with Grafana + Loki
5. Production Checklist

Estimated time: 2-3 hours
Requirements: Jetson Nano (JetPack 4.6.x), Docker, ~20GB disk
"""

# ============================================================================
# PART 1: PREPARE ML MODELS
# ============================================================================

## Step 1.1: Train CNN Autoencoder (on development machine with GPU)

### Prerequisites:
- Python 3.8+
- PyTorch 1.9+ with CUDA support
- scikit-learn, numpy, joblib

### Run on training machine:

```bash
python app/cnn/train_autoencoder.py

# Output: model saved to models/cnn_ae.pth
```

### Expected output:
```
Epoch 10/50, Loss: 0.015234
Epoch 20/50, Loss: 0.008912
...
Epoch 50/50, Loss: 0.003421
Reconstruction error threshold (p95): 0.0842
```

---

## Step 1.2: Export to ONNX

```bash
python app/cnn/export_onnx.py

# Output: models/cnn_ae.onnx
```

### Verify ONNX model:
```bash
python -c "
import onnx
model = onnx.load('models/cnn_ae.onnx')
onnx.checker.check_model(model)
print('✓ ONNX model valid')
"
```

---

## Step 1.3: Prepare RF + XGB models

Ensure the following files exist in `models/`:
- `rf_model.pkl` (RandomForest classifier)
- `xgb_model.pkl` (XGBoost classifier)
- `scaler.pkl` (StandardScaler for normalization)
- `feature_mask.npy` (Boolean array of selected features)
- `label_encoder.pkl` (Maps class indices to names)

These are trained via separate ML pipelines and should already exist.

---

## Step 1.4: Verify model files

```bash
ls -lh models/
# Expected:
# rf_model.pkl        (≈ 50 MB)
# xgb_model.pkl       (≈ 10 MB)
# scaler.pkl          (≈ 1 KB)
# feature_mask.npy    (≈ 100 B)
# label_encoder.pkl   (≈ 1 KB)
# cnn_ae.onnx         (≈ 200 KB)
```

---

# ============================================================================
# PART 2: BUILD TENSORRT ENGINE (on Jetson Nano)
# ============================================================================

## Step 2.1: Transfer ONNX to Jetson Nano

```bash
# From dev machine:
scp models/cnn_ae.onnx jetson@jetson-nano:/home/jetson/edge_fhir_hybrid/models/

# From Jetson Nano (via SSH):
ssh jetson@jetson-nano
cd /home/jetson/edge_fhir_hybrid
```

---

## Step 2.2: Build TensorRT Engine

### Create a script: `build_trt_engine.py`

```python
#!/usr/bin/env python3
"""Build TensorRT engine from ONNX on Jetson Nano."""

import tensorrt as trt
import os

def build_engine_from_onnx(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    success = parser.parse_from_file(onnx_path)
    if not success:
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20  # 1 MB for Jetson Nano
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    
    # FP16 optimization (speeds up inference ~2x)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 enabled")
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    print(f"✓ TensorRT engine saved: {engine_path}")

if __name__ == "__main__":
    onnx_path = "models/cnn_ae.onnx"
    engine_path = "models/cnn_ae.engine"
    
    print(f"Building TensorRT engine from {onnx_path}...")
    build_engine_from_onnx(onnx_path, engine_path)
```

### Run on Jetson Nano:

```bash
python3 build_trt_engine.py

# Expected output:
# Building TensorRT engine from models/cnn_ae.onnx...
# ✓ FP16 enabled
# ✓ TensorRT engine saved: models/cnn_ae.engine
```

---

## Step 2.3: Verify TensorRT engine

```bash
ls -lh models/cnn_ae.engine
# ~300-500 KB engine file
```

---

# ============================================================================
# PART 3: DOCKER BUILD & DEPLOYMENT
# ============================================================================

## Step 3.1: Build Docker image on Jetson Nano

```bash
cd /home/jetson/edge_fhir_hybrid

# Option A: Build locally (takes ~10-15 min on Jetson Nano)
docker build -t edge-fhir-hybrid:latest .

# Option B: Use pre-built image (not available yet; use Option A)
# docker pull your-registry/edge-fhir-hybrid:latest
```

---

## Step 3.2: Run container with GPU support

### Standalone Flask server:

```bash
docker run --gpus all \
  -p 5001:5001 \
  -v /home/jetson/edge_fhir_hybrid/models:/workspace/models:ro \
  -v /home/jetson/edge_fhir_hybrid/logs:/workspace/logs:rw \
  --name edge-fhir-hybrid \
  edge-fhir-hybrid:latest
```

### With docker-compose (recommended):

```bash
docker-compose -f docker-compose.grafana.yml up -d
```

### Verify containers running:

```bash
docker ps
# Expected:
# edge-fhir-hybrid   (Flask API, port 5001)
# loki               (Log storage, port 3100)
# promtail           (Log shipper)
# grafana            (Dashboard, port 3000)
```

---

## Step 3.3: Test FHIR endpoint

```bash
curl -X POST http://localhost:5001/fhir/notify \
  -H "Content-Type: application/json" \
  -d @sample_audit.json

# Expected response:
# {
#   "pred": "DDoS",
#   "score": 0.93,
#   "sev": "HIGH",
#   "anom": true,
#   "meta": {...},
#   "cnn": {"mse": 0.18, "available": true}
# }
```

---

## Step 3.4: Monitor logs

```bash
docker logs -f edge-fhir-hybrid

# Expected:
# Starting Edge FHIR Security Server on 0.0.0.0:5001
# ✓ Hybrid classifier initialized
# ✓ CNN Autoencoder loaded: models/cnn_ae.onnx
```

---

# ============================================================================
# PART 4: GRAFANA + LOKI MONITORING
# ============================================================================

## Step 4.1: Access Grafana dashboard

```
http://jetson-nano-ip:3000
Default credentials: admin / admin
```

---

## Step 4.2: Configure data source

1. Settings → Data Sources → Add Loki
2. URL: http://loki:3100
3. Save & Test

---

## Step 4.3: Import dashboard

1. Dashboards → Import
2. Upload `config/grafana/dashboards/fhir-security.json`
3. Select Loki as data source
4. View alerts in real-time

---

## Step 4.4: Set up alert rules (optional)

In Grafana:
1. Alerting → Alert Rules → New
2. Query: `{job="fhir-security"} | json | sev="HIGH"`
3. Condition: Count > 5 (per 5 minutes)
4. Notification: Email, Slack, PagerDuty, etc.

---

# ============================================================================
# PART 5: PRODUCTION CHECKLIST
# ============================================================================

- [ ] CNN Autoencoder trained on **normal** traffic only
- [ ] ONNX model exported and verified
- [ ] TensorRT engine built on **target Jetson device**
- [ ] RF + XGB models validated
- [ ] All model files copied to container
- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] `/health` endpoint responds with 200 OK
- [ ] `/fhir/notify` classifies test data correctly
- [ ] Alerts logged to `alerts.log` (JSONL format)
- [ ] Grafana dashboard displays alerts
- [ ] CNN inference latency < 50ms per sample
- [ ] CPU usage < 80% (Jetson Nano has limited resources)
- [ ] Memory usage < 4GB
- [ ] No GPU memory leaks observed over 24h

---

# ============================================================================
# PART 6: TROUBLESHOOTING
# ============================================================================

## Issue: "TensorRT not found"

```bash
# Solution: Use correct base image
# Must be: nvcr.io/nvidia/l4t-ml:r32.6.1-py3
# (includes TensorRT 8.2.0)
```

## Issue: "CUDA out of memory"

```bash
# Reduce batch size or enable INT8 quantization
# Edit docker-compose.grafana.yml:
# environment:
#   - CUDA_DEVICE_ORDER=PCI_BUS_ID
#   - CUDA_VISIBLE_DEVICES=0
```

## Issue: "CNN model not found"

```bash
# Ensure models/cnn_ae.onnx or models/cnn_ae.engine exists
# Server falls back to RF/XGB only (still functional, but less sensitive)
```

## Issue: "Feature dimension mismatch"

```python
# Check app/cnn/train_autoencoder.py:
# input_dim = 25 (must match extract_features() output)

# And app/server.py:
# reshape_for_cnn(features) # Reshapes to (1, 1, 25, 1)
```

---

# ============================================================================
# PART 7: PERFORMANCE OPTIMIZATION
# ============================================================================

## Latency targets for Jetson Nano:

```
Feature extraction:  5-10 ms
RF/XGB inference:    10-20 ms
CNN TensorRT:        15-30 ms
JSON serialization:  2-5 ms
Total per request:   ~50 ms
```

## If too slow:

1. Enable INT8 quantization in TensorRT builder
2. Reduce CNN model size (fewer layers)
3. Use ONNX Runtime with CPU threading optimization

## Monitoring inference time:

Add to `app/server.py`:

```python
import time

@fhir_notify()
def fhir_notify():
    start = time.time()
    # ... inference ...
    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"Request latency: {elapsed_ms:.1f} ms")
```

---

# ============================================================================
# PART 8: SCALING TO MULTIPLE JETSON DEVICES
# ============================================================================

For hospital deployment (multiple edge nodes):

1. Use a shared Grafana instance (cloud or central server)
2. Forward logs from each Jetson to central Loki
3. Add device labels: `device: "jetsonnano-01"`

```yaml
# config/promtail.yaml
labels:
  device: "jetsonnano-01"
  location: "ICU-floor-3"
  client: "hospital-xyz"
```

---

# ============================================================================
# NEXT STEPS
# ============================================================================

1. ✅ Train CNN Autoencoder
2. ✅ Export to ONNX
3. ✅ Build Docker image
4. ✅ Deploy to Jetson Nano
5. ✅ Verify inference accuracy
6. ⏳ Integrate with hospital FHIR server
7. ⏳ Set up 24/7 monitoring
8. ⏳ Tune alert thresholds based on real data
9. ⏳ Document SOP for hospital staff
10. ⏳ Plan disaster recovery

---

Questions? Check app/cnn/trt_runtime.py for CNN inference details.
