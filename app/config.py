import os

# Allow overriding paths via environment for local development.
# Default paths are the production locations inside the container.
MODELS_DIR = os.getenv("MODELS_DIR", "/opt/app/models")
LOG_FILE = os.getenv("LOG_FILE", "/opt/app/logs/alerts.log")

NORMAL_CLASS = os.getenv("NORMAL_CLASS", "Normal")
try:
	SEV_HIGH = float(os.getenv("SEV_HIGH", "0.95"))
	SEV_MED = float(os.getenv("SEV_MED", "0.85"))
except ValueError:
	SEV_HIGH = 0.95
	SEV_MED = 0.85
