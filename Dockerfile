# Production-grade Dockerfile for Edge FHIR Hybrid on NVIDIA Jetson Nano.
#
# Base image: nvcr.io/nvidia/l4t-ml (includes CUDA 10.2 + cuDNN + TensorRT 8.x)
# Target: JetPack 4.6.x (Jetson Nano)
# GPU: NVIDIA Maxwell (5.3) architecture
#
# Features:
# - TensorRT 8.x for CNN Autoencoder inference
# - CUDA 10.2 runtime
# - Python 3.6 + ML stack (PyTorch, TensorFlow, scikit-learn)
# - Optimized for edge inference (~3.7 GB image)
#
# Build on Jetson Nano:
#   docker build -t edge-fhir-hybrid:latest .
#
# Run:
#   docker run --gpus all -p 5001:5001 edge-fhir-hybrid:latest

# ===== BASE IMAGE (includes TensorRT + CUDA) =====
FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

# Metadata
LABEL maintainer="FHIR Security Team"
LABEL description="Edge FHIR Hybrid ML - CNN + RF/XGB on Jetson Nano"

# ===== ENVIRONMENT =====
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    TRT_RELEASE=8.2.0 \
    JETPACK_VERSION=4.6.1

# ===== DEPENDENCIES =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    # Python runtime
    python3-dev \
    python3-pip \
    # System libraries
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    # Networking
    netcat \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# ===== PYTHON PACKAGES =====
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ===== TENSORRT PLUGINS (already in base image, but ensure availability) =====
# TensorRT shared libraries should be in:
# /usr/lib/aarch64-linux-gnu/libnvonnxparser.so
# /usr/lib/aarch64-linux-gnu/libnvinfer.so
# This is included in nvcr.io/nvidia/l4t-ml image

# ===== APPLICATION CODE =====
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create required directories
RUN mkdir -p /workspace/logs \
    && mkdir -p /workspace/models \
    && mkdir -p /workspace/config \
    && chmod -R 755 /workspace

# ===== HEALTH CHECK =====
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# ===== NVIDIA RUNTIME (must use --gpus all at run time) =====
# No explicit setup needed; Docker runtime handles this.
# Ensure GPU access:
#   - Run: docker run --gpus all ...
#   - Or in docker-compose: runtime: nvidia

# ===== ENTRYPOINT =====
EXPOSE 5001

# Start Flask server with proper Python path
CMD ["python3", "-m", "app.server"]
