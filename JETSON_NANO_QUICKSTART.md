# Jetson Nano Quick Start Cheat Sheet

## TL;DR - Get Running in 5 Minutes (if Docker already installed)

```bash
# 1. Clone repo
git clone https://github.com/Hasan8936/edge_fhir_hybrid.git
cd edge_fhir_hybrid

# 2. Generate test models
python3 generate_dummy_models.py

# 3. Build and run
docker-compose up --build -d

# 4. Test
curl http://127.0.0.1:5001/health
```

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **Device** | NVIDIA Jetson Nano (any revision) |
| **OS** | JetPack 4.x (Ubuntu 18.04) |
| **RAM** | 2GB minimum, 4GB recommended |
| **Storage** | 20GB free space on microSD/eMMC |
| **Network** | Ethernet or WiFi |

---

## Full Installation Steps

### 1. Flash JetPack Image to microSD

```bash
# Download from:
# https://developer.nvidia.com/jetson-nano-sd-card-image

# Flash using Balena Etcher:
# 1. Open Etcher
# 2. Select .img file
# 3. Select microSD card
# 4. Click Flash
# 5. Wait 15 min
```

### 2. Boot and Initial Setup

```bash
# Insert microSD, connect peripherals, power on
# Follow on-screen wizard
# Update system:
sudo apt-get update
sudo apt-get upgrade -y
```

### 3. Install Docker

```bash
# Add Docker repo
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=arm64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Install
sudo apt-get update
sudo apt-get install -y docker-ce docker-compose

# Add user to group
sudo usermod -aG docker $USER
newgrp docker

# Test
docker --version
```

### 4. Clone and Setup

```bash
mkdir ~/edge_fhir && cd ~/edge_fhir
git clone https://github.com/Hasan8936/edge_fhir_hybrid.git
cd edge_fhir_hybrid
```

### 5. Generate Models

```bash
python3 generate_dummy_models.py
# Or copy your own models to models/
```

### 6. Build and Deploy

```bash
docker-compose build   # Takes 10-20 min on Jetson
docker-compose up -d   # Start service
docker-compose logs edge_node  # View logs
```

---

## Testing

### Health Check
```bash
curl http://127.0.0.1:5001/health
```

### Send Test Event
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"resourceType":"AuditEvent","action":"E","outcome":0,"agent":[{"userId":"test","network":{"address":"192.168.1.1"}}],"event":{"type":{"code":"login"}}}' \
  http://127.0.0.1:5001/fhir/notify
```

### View Alerts
```bash
cat logs/alerts.log
tail -f logs/alerts.log  # Watch in real-time
```

---

## Docker Commands

| Command | Purpose |
|---------|---------|
| `docker-compose build` | Build image |
| `docker-compose up -d` | Start in background |
| `docker-compose down` | Stop and remove |
| `docker-compose logs edge_node` | View logs |
| `docker-compose logs -f edge_node` | Follow logs (Ctrl+C to exit) |
| `docker-compose restart edge_node` | Restart container |
| `docker-compose ps` | Show running containers |
| `docker exec -it edge_node /bin/bash` | SSH into container |

---

## Jetson Useful Commands

| Command | Purpose |
|---------|---------|
| `hostname -I` | Get Jetson IP address |
| `sudo tegrastats` | View CPU/GPU/RAM in real-time (Ctrl+C to exit) |
| `docker stats` | View container resource usage |
| `free -h` | Check memory |
| `df -h` | Check disk space |

---

## Configure FHIR Server

Get Jetson IP:
```bash
hostname -I
# Example: 192.168.1.50
```

Create Subscription on FHIR Server:
```json
{
  "resourceType": "Subscription",
  "status": "active",
  "criteria": "AuditEvent",
  "channel": {
    "type": "rest-hook",
    "endpoint": "http://192.168.1.50:5001/fhir/notify"
  }
}
```

---

## Troubleshooting

### Problem: Docker command fails
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Problem: Port 5001 already in use
```bash
# Stop the service
docker-compose down

# Or use different port in docker-compose.yml
```

### Problem: Model artifacts not found
```bash
# Generate dummy models
python3 generate_dummy_models.py

# Or copy your models
scp rf_model.pkl xgb_model.pkl scaler.pkl feature_mask.npy label_encoder.pkl \
    jetson@JETSON_IP:~/edge_fhir/edge_fhir_hybrid/models/
```

### Problem: Out of memory
```bash
# Add swap (4GB)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## File Structure

```
edge_fhir_hybrid/
├── app/                          # Python source code
│   ├── __init__.py
│   ├── server.py                # Flask API
│   ├── edge_model.py            # Model inference
│   ├── detector.py              # Anomaly detection
│   ├── fhir_features.py         # FHIR → features
│   └── config.py                # Configuration
├── models/                       # Model artifacts (YOU provide these)
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_mask.npy
│   └── label_encoder.pkl
├── logs/                         # Alert logs (auto-created)
│   └── alerts.log
├── config/                       # Example configurations
├── tools/                        # Utilities
├── Dockerfile                    # Docker build config
├── docker-compose.yml            # Container orchestration
├── requirements.txt              # Python dependencies
├── generate_dummy_models.py      # Test model generator
├── README.md                     # Main documentation
├── DEPLOYMENT.md                 # Deployment guide
├── JETSON_NANO_SETUP.md         # This file
└── JETSON_NANO_QUICKSTART.md    # Quick reference
```

---

## Performance Tips

- **Enable max performance mode:**
  ```bash
  sudo nvpmodel -m 0     # Max performance
  sudo nvpmodel -q       # Check current mode
  ```

- **Monitor resources:**
  ```bash
  sudo tegrastats        # Real-time stats
  docker stats           # Container resources
  ```

- **Optimize for inference:**
  - Use TensorRT for model acceleration (advanced)
  - Batch multiple events before inference
  - Consider reducing model ensemble weights

---

## Network Configuration

### Local Network (Development)
```bash
# Everything on same network, no special setup needed
curl http://192.168.1.50:5001/health
```

### Internet/WAN (Production)
```bash
# DO NOT expose directly! Use reverse proxy:
# FHIR Server → HTTPS → NGINX (TLS) → Jetson (internal)
```

---

## Keep Your System Updated

```bash
# Update OS packages
sudo apt-get update && sudo apt-get upgrade -y

# Restart after major updates
sudo reboot

# Check JetPack version
apt-cache show nvidia-jetpack
```

---

## Emergency Commands

```bash
# If container is stuck
docker kill edge_node
docker-compose down

# If you need to start fresh
docker-compose down -v
docker image rm edge-fhir-hybrid:latest
docker-compose up --build -d

# Full cleanup (WARNING: removes all data)
docker-compose down -v
rm -rf logs/alerts.log
```

---

## Success Indicators

- ✅ `docker-compose ps` shows container UP
- ✅ `curl http://127.0.0.1:5001/health` returns `{"status":"ok"}`
- ✅ `curl http://127.0.0.1:5001/config` returns model config
- ✅ `logs/alerts.log` exists and receives events
- ✅ FHIR server can POST to Jetson IP:5001

---

## Next Steps

1. Test with real FHIR events from your server
2. Monitor `logs/alerts.log` for anomalies
3. Integrate with your SIEM (Splunk, ELK, etc.)
4. Set up TLS/HTTPS via API gateway
5. Consider redundancy (multiple Jetson nodes)

---

## Support

- **Issues?** Check `docker-compose logs edge_node`
- **Questions?** See full guide: `JETSON_NANO_SETUP.md`
- **Documentation:** https://github.com/Hasan8936/edge_fhir_hybrid
