FROM nvcr.io/nvidia/l4t-base:r32.7.1

# Install minimal Python runtime and required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# Use python -m pip to ensure the right interpreter is used
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/app
COPY app/ /opt/app/
COPY config/ /opt/app/config/
RUN mkdir -p /opt/app/logs

EXPOSE 5001
# Run the Flask app via the package entrypoint
CMD ["python3", "-m", "app.server"]
