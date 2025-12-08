#!/bin/bash
# Pre-flight checklist for Jetson Nano edge_fhir_hybrid deployment

set -e

echo "=========================================="
echo "  JETSON NANO PRE-FLIGHT CHECKLIST"
echo "=========================================="
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Helper functions
check_ok() {
    echo "  ✓ $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo "  ✗ $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo "  ⚠ $1"
    ((WARNINGS++))
}

# 1. Check OS
echo "[1/9] Checking OS and JetPack..."
if grep -q "jetson" /etc/os-release; then
    check_ok "Running on Jetson OS"
else
    check_warn "Not detected as Jetson (but may still work)"
fi

if grep -q "Ubuntu" /etc/os-release; then
    check_ok "Ubuntu OS detected"
else
    check_fail "Not Ubuntu"
fi

# 2. Check Docker
echo ""
echo "[2/9] Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    check_ok "Docker installed: $DOCKER_VERSION"
else
    check_fail "Docker not installed. Run: sudo apt-get install docker-ce"
fi

if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    check_ok "Docker Compose installed: $COMPOSE_VERSION"
else
    check_fail "Docker Compose not installed. Run: sudo apt-get install docker-compose"
fi

# 3. Check Docker daemon
echo ""
echo "[3/9] Checking Docker daemon..."
if docker ps &> /dev/null; then
    check_ok "Docker daemon is running"
else
    check_fail "Docker daemon not accessible. Run: sudo systemctl start docker"
fi

# 4. Check file permissions
echo ""
echo "[4/9] Checking permissions..."
if groups $USER | grep -q docker; then
    check_ok "User is in docker group"
else
    check_warn "User not in docker group. Run: sudo usermod -aG docker \$USER && newgrp docker"
fi

# 5. Check required directories
echo ""
echo "[5/9] Checking repository structure..."
if [ -f "docker-compose.yml" ]; then
    check_ok "docker-compose.yml found"
else
    check_fail "docker-compose.yml not found. Run: git clone ... && cd edge_fhir_hybrid"
fi

if [ -d "app" ]; then
    check_ok "app/ directory found"
else
    check_fail "app/ directory not found"
fi

if [ -d "models" ]; then
    check_ok "models/ directory found"
else
    check_fail "models/ directory not found"
fi

# 6. Check model artifacts
echo ""
echo "[6/9] Checking model artifacts..."
MODEL_FILES=("rf_model.pkl" "xgb_model.pkl" "scaler.pkl" "feature_mask.npy" "label_encoder.pkl")
MODELS_FOUND=0

for model in "${MODEL_FILES[@]}"; do
    if [ -f "models/$model" ]; then
        check_ok "models/$model found"
        ((MODELS_FOUND++))
    fi
done

if [ $MODELS_FOUND -eq 0 ]; then
    check_warn "No model artifacts found in models/. Run: python3 generate_dummy_models.py"
elif [ $MODELS_FOUND -lt 5 ]; then
    check_warn "Only $MODELS_FOUND/5 model artifacts found"
fi

# 7. Check system resources
echo ""
echo "[7/9] Checking system resources..."
MEMORY_GB=$(free -h | awk '/^Mem:/ {print $2}')
MEMORY_AVAILABLE=$(free -h | awk '/^Mem:/ {print $7}')
check_ok "Total RAM: $MEMORY_GB (available: $MEMORY_AVAILABLE)"

DISK_AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
DISK_FREE=$(df -h . | awk 'NR==2 {print $3}')
check_ok "Disk space: $DISK_FREE used, $DISK_AVAILABLE available"

# Check swap
if [ -n "$(swapon --show)" ]; then
    SWAP_SIZE=$(free -h | awk '/^Swap:/ {print $2}')
    check_ok "Swap enabled: $SWAP_SIZE"
else
    check_warn "No swap configured (optional but recommended)"
fi

# 8. Check network
echo ""
echo "[8/9] Checking network..."
if hostname -I &> /dev/null; then
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
    check_ok "Network IP: $IP_ADDRESS"
else
    check_fail "Cannot determine IP address"
fi

if ping -c 1 8.8.8.8 &> /dev/null; then
    check_ok "Internet connectivity verified"
else
    check_warn "Cannot ping 8.8.8.8 (may be firewalled, but local network should work)"
fi

# 9. Check Python and dependencies
echo ""
echo "[9/9] Checking Python and packages..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    check_ok "$PYTHON_VERSION found"
else
    check_fail "Python3 not found"
fi

if [ -f "requirements.txt" ]; then
    check_ok "requirements.txt found"
    # Check if key packages can be imported (don't install, just check)
    if python3 -c "import flask" 2>/dev/null; then
        check_ok "Flask is installed"
    else
        check_warn "Flask not installed (will be installed in Docker)"
    fi
else
    check_fail "requirements.txt not found"
fi

# Summary
echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
echo "✓ Checks Passed: $CHECKS_PASSED"
echo "✗ Checks Failed: $CHECKS_FAILED"
echo "⚠ Warnings: $WARNINGS"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ System is ready! You can proceed with:"
    echo "   docker-compose build"
    echo "   docker-compose up -d"
else
    echo "⚠️  Please fix the $CHECKS_FAILED failed check(s) above."
    exit 1
fi

if [ $WARNINGS -gt 0 ]; then
    echo ""
    echo "Note: $WARNINGS warning(s) detected. See above for details."
    echo "These are typically not blocking, but may need attention."
fi

echo ""
echo "Next steps:"
echo "1. docker-compose build          (build Docker image)"
echo "2. docker-compose up -d          (start service)"
echo "3. curl http://127.0.0.1:5001/health  (test service)"
echo ""
