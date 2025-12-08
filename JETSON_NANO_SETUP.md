# Jetson Nano Implementation Guide - Step by Step

This guide walks you through implementing the edge FHIR hybrid ML service **directly on a Jetson Nano** from start to finish.

## Table of Contents

1. [Prerequisites & Hardware Setup](#prerequisites--hardware-setup)
2. [Initial Jetson Setup](#initial-jetson-setup)
3. [Install Docker](#install-docker)
4. [Prepare Your Jetson](#prepare-your-jetson)
5. [Clone the Repository](#clone-the-repository)
6. [Prepare Model Artifacts](#prepare-model-artifacts)
7. [Build & Run](#build--run)
8. [Test the Service](#test-the-service)
9. [Configure FHIR Server](#configure-fhir-server)
10. [Monitor & Maintain](#monitor--maintain)

---

## Prerequisites & Hardware Setup

### What You Need

- **NVIDIA Jetson Nano 4GB or 2GB** (any revision)
- **Power supply:** 5V/4A USB-C (recommended) or barrel connector
- **microSD card:** 64GB or larger, Class A1 or A2
- **USB keyboard & mouse** (for initial setup)
- **HDMI monitor** (for initial setup)
- **Ethernet cable** (recommended) or WiFi dongle
- **Computer with:** 
  - Etcher or Balena Etcher (for flashing microSD)
  - SSH client (PuTTY, Windows Terminal, or Mac Terminal)

### Hardware Connections

```
┌─────────────────────────┐
│   JETSON NANO           │
├─────────────────────────┤
│ USB-C (Power) ────────→ Power Adapter (5V/4A)
│ HDMI ──────────────────→ Monitor
│ Ethernet ──────────────→ Network/Router
│ USB-A ─────────────────→ Keyboard + Mouse (via hub)
└─────────────────────────┘
```

---

## Initial Jetson Setup

### Step 1.1: Flash JetPack to microSD Card

**On your computer (Windows/Mac/Linux):**

1. Download **JetPack 4.6.x** from:
   - https://developer.nvidia.com/jetson-nano-sd-card-image
   - Get the `.zip` file (~6 GB)

2. Extract the `.zip` file → you'll get an `.img` file

3. **Flash to microSD using Balena Etcher:**
   - Download: https://www.balena.io/etcher/
   - Open Etcher
   - Select image: the `.img` file you extracted
   - Select target: your microSD card reader
   - Click "Flash"
   - Wait 10-15 minutes

4. Eject the microSD card safely from your computer

### Step 1.2: Boot Jetson for the First Time

1. Insert the flashed microSD card into the **Jetson Nano's microSD slot**
2. Connect keyboard, mouse, HDMI monitor, ethernet, and power
3. **Power on** the Jetson
4. Follow the on-screen setup wizard:
   - Select language and timezone
   - Create a user account (e.g., `jetson` / password `jetson`)
   - Accept NVIDIA license
   - Wait for system to finish updating (may take 5+ minutes)

5. **Reboot** when prompted

### Step 1.3: Connect to Network

1. Open a terminal on the Jetson (right-click desktop → "Open Terminal")
2. Check your IP address:
   ```bash
   hostname -I
   ```
   Example output: `192.168.1.50`
   **Note this IP address** — you'll need it later.

3. Update system packages:
   ```bash
   sudo apt-get update
   sudo apt-get upgrade -y
   ```
   This may take 10+ minutes.

---

## Install Docker

### Step 2.1: Add Docker Repository

```bash
# Add Docker's GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository (JetPack uses Ubuntu 18.04)
sudo add-apt-repository \
   "deb [arch=arm64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

### Step 2.2: Install Docker & Docker Compose

```bash
# Update package list
sudo apt-get update

# Install Docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
sudo apt-get install -y docker-compose

# Verify installation
docker --version
docker-compose --version
```

### Step 2.3: Add Your User to Docker Group

So you don't need `sudo` every time:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Activate the new group membership
newgrp docker

# Test (should not require sudo)
docker ps
```

---

## Prepare Your Jetson

### Step 3.1: Create Working Directory

```bash
# Create a directory for the edge service
mkdir -p ~/edge_fhir
cd ~/edge_fhir
```

### Step 3.2: Increase Swap (Optional but Recommended)

The Jetson Nano has limited RAM (2GB or 4GB). Increasing swap helps during model training/testing:

```bash
# Check current swap
free -h

# Create 4GB swap file (takes a minute)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

---

## Clone the Repository

### Step 4.1: Clone from GitHub

```bash
cd ~/edge_fhir
git clone https://github.com/Hasan8936/edge_fhir_hybrid.git
cd edge_fhir_hybrid
```

### Step 4.2: Verify Files

```bash
# List the repository structure
ls -la

# You should see:
# - app/             (Python source code)
# - models/          (where model artifacts will go)
# - logs/            (where alerts will be logged)
# - config/          (FHIR configuration examples)
# - Dockerfile       (Docker build configuration)
# - docker-compose.yml
# - requirements.txt
# - README.md
```

---

## Prepare Model Artifacts

### Option A: Generate Dummy Models (For Testing)

If you don't have trained models yet, use dummy models to test the system:

```bash
cd ~/edge_fhir/edge_fhir_hybrid

# Run the dummy model generator
python3 generate_dummy_models.py
```

Expected output:
```
Creating dummy model artifacts...
✓ models/rf_model.pkl
✓ models/xgb_model.pkl
✓ models/scaler.pkl
✓ models/feature_mask.npy
✓ models/label_encoder.pkl

✅ All dummy model artifacts created successfully!
   Classes: ['Normal', 'Attack']
   Features: 8
```

Verify files were created:

```bash
ls -lh models/

# You should see:
# -rw-r--r-- 1 jetson jetson 1.2K rf_model.pkl
# -rw-r--r-- 1 jetson jetson 1.1K xgb_model.pkl
# -rw-r--r-- 1 jetson jetson 2.5K scaler.pkl
# -rw-r--r-- 1 jetson jetson  100B feature_mask.npy
# -rw-r--r-- 1 jetson jetson  150B label_encoder.pkl
```

### Option B: Copy Your Trained Models

If you have a trained model from your development machine:

```bash
# On your development machine, zip the model files:
zip models.zip rf_model.pkl xgb_model.pkl scaler.pkl feature_mask.npy label_encoder.pkl

# Copy to Jetson (from your dev machine, replace JETSON-IP):
scp models.zip jetson@JETSON-IP:~/edge_fhir/edge_fhir_hybrid/

# On Jetson, extract:
cd ~/edge_fhir/edge_fhir_hybrid
unzip -o models.zip -d models/
rm models.zip

# Verify
ls -lh models/
```

---

## Build & Run

### Step 5.1: Build the Docker Image

This may take 10-20 minutes on the Jetson (it's slower than a desktop):

```bash
cd ~/edge_fhir/edge_fhir_hybrid

# Build the Docker image
docker-compose build

# Expected output (last few lines):
# ...
# Step 20/20 : CMD ["python3", "-m", "app.server"]
# ---> Running in abc123def456
# ---> Successfully built xyz789
# Successfully tagged edge-fhir-hybrid:latest
```

### Step 5.2: Start the Service

```bash
# Start the service in the background
docker-compose up -d

# Expected output:
# Creating edge_node ... done
```

### Step 5.3: Verify the Container is Running

```bash
# Check container status
docker-compose ps

# Expected output:
# NAME      IMAGE                    STATUS
# edge_node edge-fhir-hybrid:latest  Up 2 seconds

# View logs
docker-compose logs edge_node

# Expected output (should show model loading):
# Creating edge_node ... done
# Attaching to edge_node
# [INFO] Loading RandomForest model from /opt/app/models/rf_model.pkl
# [INFO] Loading XGBoost model from /opt/app/models/xgb_model.pkl
# ...
#  * Running on http://0.0.0.0:5001
```

---

## Test the Service

### Step 6.1: Health Check

Test that the service is responding:

```bash
curl http://127.0.0.1:5001/health
```

Expected response:
```json
{"status":"ok"}
```

### Step 6.2: Get Configuration

```bash
curl http://127.0.0.1:5001/config
```

Expected response:
```json
{
  "models_dir": "/opt/app/models",
  "normal_class": "Normal",
  "classes": ["Normal", "Attack"]
}
```

### Step 6.3: Send a Test FHIR Event

```bash
# Create a test FHIR event
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "resourceType": "AuditEvent",
    "action": "E",
    "outcome": 0,
    "agent": [
      {
        "userId": "user123",
        "network": {"address": "192.168.1.100"}
      }
    ],
    "event": {"type": {"code": "login"}}
  }' \
  http://127.0.0.1:5001/fhir/notify
```

Expected response:
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

### Step 6.4: Check Alert Logs

```bash
# View alerts (if any anomalies were detected)
cat logs/alerts.log

# Watch for new alerts in real-time
tail -f logs/alerts.log
```

---

## Configure FHIR Server

### Step 7.1: Register Subscription on Your FHIR Server

If your FHIR server is HAPI FHIR, Azure API for FHIR, or similar, create a Subscription resource:

**Get your Jetson's IP address:**
```bash
hostname -I
# Example: 192.168.1.50
```

**Create a Subscription in your FHIR server (via POST or web UI):**

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

Replace `192.168.1.50` with your actual Jetson IP.

### Step 7.2: Test the Subscription

On your FHIR server:
- Create a new `AuditEvent` resource
- The FHIR server should POST it to your Jetson
- Check Jetson logs for the event

```bash
# On Jetson, watch the logs
docker-compose logs -f edge_node
```

---

## Monitor & Maintain

### Step 8.1: View Logs

```bash
# View service logs
docker-compose logs edge_node

# Follow logs in real-time
docker-compose logs -f edge_node

# View only the last 50 lines
docker-compose logs --tail 50 edge_node
```

### Step 8.2: View Alerts

```bash
# View all alerts
cat logs/alerts.log

# View only alerts from the last hour
find logs/ -name "alerts.log" -mmin -60 | xargs tail

# Count anomalies
grep '"anom": true' logs/alerts.log | wc -l
```

### Step 8.3: Stop the Service

```bash
# Stop the container
docker-compose down

# Container will be removed but images and volumes remain
```

### Step 8.4: Restart the Service

```bash
# Just restart (if already built)
docker-compose up -d

# Or rebuild + start
docker-compose up --build -d
```

### Step 8.5: Clean Up

```bash
# Remove stopped containers
docker-compose down -v

# Remove unused Docker images (careful!)
docker image prune -a

# Clean logs (keep alerts, just clear old ones)
echo "" > logs/alerts.log
```

---

## Troubleshooting Common Issues

### Issue: "Permission denied" when accessing docker

**Fix:** Add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "Could not connect to Docker daemon"

**Fix:** Ensure Docker is running:
```bash
sudo systemctl start docker
sudo systemctl enable docker  # Start on boot
```

### Issue: Container exits immediately / "Model artifacts not found"

**Fix:** Ensure model files exist:
```bash
ls -la models/
# Should show: rf_model.pkl, xgb_model.pkl, scaler.pkl, feature_mask.npy, label_encoder.pkl

# If missing, generate dummy models:
python3 generate_dummy_models.py
```

### Issue: "Port 5001 already in use"

**Fix:** Either stop the existing service or use a different port:
```bash
# Option 1: Stop the existing service
docker-compose down

# Option 2: Use a different port
# Edit docker-compose.yml and change "ports: 5001:5001" to "ports: 5002:5001"
```

### Issue: High CPU/Memory usage

**Fix:** The Jetson Nano has limited resources. This is normal, but you can:
- Check resource usage: `docker stats`
- Reduce batch size of FHIR events
- Add swap (see Step 3.2)
- Use async processing with a queue

### Issue: "Cannot connect to Jetson from FHIR server"

**Fix:** Check network connectivity:
```bash
# From Jetson, check IP address
hostname -I

# From your FHIR server machine, test connectivity
ping 192.168.1.50  # Replace with Jetson IP

# Test the service
curl http://192.168.1.50:5001/health
```

---

## Quick Reference Commands

### Common Commands on Jetson

```bash
# Check Jetson's current IP
hostname -I

# Check disk space
df -h

# Check memory usage
free -h

# Check CPU temperature
sudo tegrastats

# Monitor Docker container in real-time
docker stats

# View Docker logs
docker logs -f edge_node

# Restart the service
docker-compose restart edge_node

# Stop everything
docker-compose down

# Start everything
docker-compose up -d

# Rebuild and start
docker-compose up --build -d

# SSH into container for debugging
docker exec -it edge_node /bin/bash
```

---

## Next Steps

1. **Test with real FHIR events:** Integrate with your actual FHIR server
2. **Monitor alerts:** Set up log aggregation (ELK, Splunk, etc.)
3. **Automate alerts:** Connect to your SIEM/SOAR system
4. **Harden security:** Use TLS/HTTPS via an API gateway
5. **Scale:** If needed, add multiple Jetson nodes or use Kubernetes

---

## Support & Resources

- **NVIDIA Jetson Nano Docs:** https://docs.nvidia.com/jetson/l4t/
- **Docker Documentation:** https://docs.docker.com/
- **FHIR REST Hooks:** https://hl7.org/fhir/subscription.html
- **GitHub Repository:** https://github.com/Hasan8936/edge_fhir_hybrid

---

## Success Checklist

- [x] Jetson Nano booted and connected to network
- [x] Docker and Docker Compose installed
- [x] Repository cloned
- [x] Model artifacts in `models/` directory
- [x] Docker image built
- [x] Service running (`docker-compose up -d`)
- [x] `/health` endpoint responds
- [x] Test FHIR event processed successfully
- [x] Alerts logged to `logs/alerts.log`
- [x] FHIR server subscription configured

Once you check all boxes, you're ready for production! 🚀
