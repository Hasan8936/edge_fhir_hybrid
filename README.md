# Edge FHIR Hybrid ML Security Node

This repository contains an edge inference service designed to run on an NVIDIA Jetson Nano (JetPack/L4T). The node receives FHIR REST-hook notifications (typically `AuditEvent` resources), extracts numeric features from the FHIR JSON, runs a pre-trained hybrid ML model (RandomForest + XGBoost) for anomaly detection/classification, and logs alerts locally.

## Architecture

```
FHIR Server 
    |
    | REST-hook POST (AuditEvent)
    v
Jetson Nano :5001/fhir/notify
    |
    +-- extract_features() -> numeric vector
    |
    +-- HybridDeployedModel.predict() -> probabilities
    |
    +-- EdgeDetector.analyze() -> {pred, score, severity, anomaly}
    |
    +-- log alerts to logs/alerts.log (if anomalous)
```

## Prerequisites

- **Jetson Nano** with JetPack 4.x (L4T base image)
- **Docker** and **Docker Compose** installed on Jetson
- **Pre-trained model artifacts** (trained offline, exported as `.pkl` and `.npy` files)

## Model Artifacts

Before running the container, you must place the following files in the `models/` directory:

- `rf_model.pkl` - Trained RandomForest model (scikit-learn)
- `xgb_model.pkl` - Trained XGBoost model
- `scaler.pkl` - StandardScaler or similar feature scaler (joblib)
- `feature_mask.npy` - Boolean array indicating selected features (NumPy)
- `label_encoder.pkl` - LabelEncoder for class labels (scikit-learn)

These files come from an offline training pipeline (not part of this repository).

## Quick Start

### 1. Place Model Artifacts

```bash
# Copy trained model files into the models/ directory
cp /path/to/trained/models/*.pkl models/
cp /path/to/trained/models/*.npy models/
```

### 2. Build and Run

```bash
cd edge_fhir_hybrid
docker-compose up --build -d
```

### 3. Verify Service is Running

```bash
# Check health
curl http://<JETSON-IP>:5001/health

# Get configuration info
curl http://<JETSON-IP>:5001/config
```

### 4. Send a Test FHIR Event

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "resourceType": "AuditEvent",
    "action": "E",
    "outcome": 0,
    "agent": [{"userId": "user123", "network": {"address": "192.168.1.100"}}],
    "event": {"type": {"code": "login"}}
  }' \
  http://<JETSON-IP>:5001/fhir/notify
```

### 5. Check Logs

```bash
# View alert log (JSON-lines format)
tail -f logs/alerts.log

# View container logs
docker-compose logs -f edge_node
```

## API Endpoints

### GET /health

Simple health check.

**Response:**
```json
{"status": "ok"}
```

### GET /config

Returns non-sensitive runtime configuration.

**Response:**
```json
{
  "models_dir": "/opt/app/models",
  "normal_class": "Normal",
  "classes": ["Normal", "Attack", "Suspicious"]
}
```

### POST /fhir/notify

Main inference endpoint. Accepts a FHIR resource (usually `AuditEvent`) and returns predictions.

**Request:**
```json
{
  "resourceType": "AuditEvent",
  "action": "E",
  "outcome": 0,
  "agent": [{"userId": "user123", "network": {"address": "192.168.1.100"}}],
  "event": {"type": {"code": "login"}}
}
```

**Response:**
```json
{
  "pred": "Normal",
  "score": 0.15,
  "sev": "LOW",
  "anom": false,
  "meta": {
    "resourceType": "AuditEvent",
    "action": "E",
    "outcome": 0.0,
    "user_hash": 1234,
    "ip_hash": 5678,
    "agent_count": 1,
    "remote_addr": "192.168.1.50"
  }
}
```

**Fields:**
- `pred`: Predicted class label (e.g., "Normal", "Attack")
- `score`: Anomaly score in [0, 1]. Score = 1 - P(normal_class)
- `sev`: Severity level ("LOW", "MEDIUM", "HIGH")
- `anom`: Boolean indicating if this is an anomalous event
- `meta`: Minimal metadata extracted from the FHIR resource (hashed for privacy)

## Configuration

All configuration values can be overridden via environment variables or modified in `app/config.py`:

- `MODELS_DIR`: Path to model artifacts inside container (default: `/opt/app/models`)
- `LOG_DIR`: Path to logs directory (default: `/opt/app/logs`)
- `LOG_FILE`: Full path to alert log (default: `/opt/app/logs/alerts.log`)
- `NORMAL_CLASS`: Label representing the normal/benign class (default: `"Normal"`)
- `SEV_HIGH`: Anomaly score threshold for HIGH severity (default: `0.95`)
- `SEV_MED`: Anomaly score threshold for MEDIUM severity (default: `0.85`)

### Setting Environment Variables in Docker Compose

```yaml
services:
  edge_node:
    environment:
      - SEV_HIGH=0.90
      - SEV_MED=0.80
```

## Feature Extraction

The `extract_features()` function converts FHIR JSON into an 8-dimensional numeric vector:

1. **resourceType_hash** - Hash of the resource type (e.g., "AuditEvent")
2. **action_hash** - Hash of the action code
3. **event_type_code_hash** - Hash of the event type code
4. **outcome_value** - Numeric outcome value (if available)
5. **user_hash** - Hash of the user ID (for privacy)
6. **ip_hash** - Hash of the agent's IP address (for privacy)
7. **agent_count** - Number of agents involved
8. **failure_flag** - Binary flag (1 if "fail" or "denied" keywords detected)

The feature vector is then scaled using the pre-trained scaler and a feature mask is applied to select only the most important features (as determined during training).

## Security Considerations

- **No PHI/PII in Logs:** The service only logs hashed metadata and prediction results, not raw FHIR payloads.
- **Model Artifacts Read-Only:** Model files are mounted as read-only inside the container.
- **TLS/Authentication:** This service should **never** be exposed directly to the internet. Always deploy behind:
  - A reverse proxy (e.g., NGINX) for TLS termination
  - An API gateway with authentication and rate limiting
  - A service mesh (e.g., Istio) for security policies

## Troubleshooting

### Service won't start / container exits immediately

Check logs:
```bash
docker-compose logs edge_node
```

Common issues:
- **Missing model artifacts:** Ensure all 5 files are in `./models/` directory before running `docker-compose up --build`.
- **Permission issues:** Ensure `./logs/` and `./models/` directories are writable.

### High latency or timeouts

The Jetson Nano has limited compute. For large batches or very frequent requests, consider:
- Pre-processing and batching events in an upstream queue (Redis, RabbitMQ)
- Tuning the XGBoost ensemble weights in `EdgeDetector`

### Model predictions seem wrong

- Verify the feature extraction logic matches your training pipeline.
- Check that `scaler.pkl`, `feature_mask.npy`, and `label_encoder.pkl` match the model artifacts.
- Review `logs/alerts.log` for metadata to understand what the model is seeing.

## Future Improvements

- **Prometheus Metrics:** Add counters for requests, anomalies, and inference latency.
- **Event Queuing:** Buffer incoming events with a lightweight queue for better throughput.
- **SIEM Integration:** Forward alerts to Splunk, ELK, or other SIEM platforms via syslog or HTTP.
- **Advanced Auth:** Implement OAuth2 or mutual TLS for secure inter-service communication.

## Development

### Running Locally (without Docker)

```bash
python3 -m pip install -r requirements.txt
python3 -m app.server
```

The server will listen on `0.0.0.0:5001`.

### Testing

Place dummy model artifacts in `models/` (or skip this step to test the 503 error handling):

```bash
# Test health
curl http://127.0.0.1:5001/health

# Test inference
curl -X POST -H "Content-Type: application/json" \
  -d '{"resourceType":"AuditEvent","action":"E","outcome":0}' \
  http://127.0.0.1:5001/fhir/notify
```

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Support

For issues or questions, please open an issue in the repository or contact the maintainers.
