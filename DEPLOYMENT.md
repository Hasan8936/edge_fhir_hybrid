# Deployment Guide for Edge FHIR Hybrid ML Service

This guide walks through deploying the edge inference service on a Jetson Nano.

## Prerequisites

- NVIDIA Jetson Nano with JetPack 4.x (L4T)
- Docker and Docker Compose installed
- A trained ML model (or use dummy models for testing)
- Network access from FHIR server to Jetson's IP:5001

## Step 1: Clone the Repository

On your Jetson Nano:

```bash
git clone https://github.com/Hasan8936/edge_fhir_hybrid.git
cd edge_fhir_hybrid
```

## Step 2: Prepare Model Artifacts

### Option A: Use Dummy Models (Testing Only)

```bash
python3 generate_dummy_models.py
```

This creates simple pickle files that satisfy the model interface but return random predictions.

### Option B: Use Your Trained Models

Copy your 5 model files into the `models/` directory:

```bash
cp /path/to/rf_model.pkl models/
cp /path/to/xgb_model.pkl models/
cp /path/to/scaler.pkl models/
cp /path/to/feature_mask.npy models/
cp /path/to/label_encoder.pkl models/
```

Verify files are present:

```bash
ls -la models/
# Expected output:
# rf_model.pkl
# xgb_model.pkl
# scaler.pkl
# feature_mask.npy
# label_encoder.pkl
```

## Step 3: Build and Run the Container

```bash
docker-compose up --build -d
```

This will:
- Build the Docker image from the Dockerfile
- Mount model artifacts (read-only) from `./models`
- Mount logs directory from `./logs`
- Start the Flask server on port 5001

Check that the container started successfully:

```bash
docker-compose logs edge_node
```

You should see:

```
[INFO] edge_fhir_hybrid.app.server - Model loaded successfully from /opt/app/models
[INFO] edge_fhir_hybrid.app.edge_model - Loading RandomForest model from /opt/app/models/rf_model.pkl
...
 * Running on http://0.0.0.0:5001
```

## Step 4: Test the Service

### Health Check

```bash
curl http://localhost:5001/health
```

Response:
```json
{"status": "ok"}
```

### Get Configuration

```bash
curl http://localhost:5001/config
```

Response:
```json
{
  "models_dir": "/opt/app/models",
  "normal_class": "Normal",
  "classes": ["Normal", "Attack"]
}
```

### Send a Test FHIR Event

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "resourceType": "AuditEvent",
    "action": "E",
    "outcome": 0,
    "agent": [{"userId": "test_user", "network": {"address": "192.168.1.100"}}],
    "event": {"type": {"code": "login"}}
  }' \
  http://localhost:5001/fhir/notify
```

Response:
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
    "user_hash": 12345,
    "ip_hash": 67890,
    "agent_count": 1,
    "remote_addr": "127.0.0.1"
  }
}
```

## Step 5: Check Alerts

When anomalies are detected, they're logged to `logs/alerts.log` (JSON-lines format):

```bash
tail -f logs/alerts.log
```

Example anomaly entry:

```json
{"ts": "2024-12-09T14:23:45.123456Z", "pred": "Attack", "score": 0.92, "sev": "HIGH", "meta": {...}}
```

## Step 6: Configure for Your FHIR Server

Update your FHIR server's subscription resource to point to your Jetson:

```json
{
  "resourceType": "Subscription",
  "status": "active",
  "criteria": "AuditEvent",
  "channel": {
    "type": "rest-hook",
    "endpoint": "http://<JETSON-IP>:5001/fhir/notify"
  }
}
```

Replace `<JETSON-IP>` with your Jetson Nano's IP address on the network.

## Troubleshooting

### Container Won't Start

Check logs:

```bash
docker-compose logs edge_node
```

Common issues:

- **"Model artifacts not available"** → Ensure model files are in `./models/`
- **"Permission denied"** → Check directory permissions: `chmod 755 models/ logs/`
- **"Port already in use"** → Change port in `docker-compose.yml` or stop other services

### High Latency

The Jetson Nano has limited compute. For high event volume:

- Pre-filter events upstream (in FHIR server or API gateway)
- Batch events before sending
- Consider async processing with a queue (Redis, etc.)

### Model Not Loading

Verify model file formats:

```bash
file models/rf_model.pkl
# Should output: "Python pickle"
```

Check that files have read permissions:

```bash
ls -l models/
```

### Predictions Seem Wrong

- Review `logs/alerts.log` to see what features the model is seeing
- Verify your model was trained on the same 8-feature vector
- Check that the `label_encoder.pkl` has the expected classes

## Maintenance

### View Logs

```bash
# Docker container logs
docker-compose logs -f edge_node

# Alert log
tail -f logs/alerts.log

# Search for errors in alert log
grep -i error logs/alerts.log
```

### Stop the Service

```bash
docker-compose down
```

### Restart the Service

```bash
docker-compose restart edge_node
```

### Update Configuration

Edit environment variables in `docker-compose.yml`:

```yaml
environment:
  - SEV_HIGH=0.90
  - SEV_MED=0.80
```

Then restart:

```bash
docker-compose up -d
```

## Security Hardening (Production)

1. **Enable TLS**: Put the service behind NGINX with SSL certificates
2. **Authentication**: Add API gateway with OAuth2 or mutual TLS
3. **Rate limiting**: Use API gateway to limit request rates
4. **Network isolation**: Use firewall rules to restrict access to trusted FHIR servers
5. **Logging retention**: Archive and rotate alert logs regularly
6. **Secrets management**: Use Docker Secrets or external vault for sensitive config

Example NGINX reverse proxy setup (not provided here):

```
- FHIR Server → HTTPS → NGINX (TLS termination) → Docker (internal network)
```

## Monitoring

For production, consider adding:

- Prometheus metrics endpoint (see `app/server.py` comments)
- Health checks in your orchestration system
- Alert aggregation to your SIEM (Splunk, ELK, etc.)
- Log shipping to centralized logging (Elastic, Stackdriver, etc.)

## Support

For issues or questions, refer to the main `README.md` or open an issue on GitHub.
