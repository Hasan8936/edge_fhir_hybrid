import os
import platform

# ---------------- PATHS ----------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
LOG_FILE = os.getenv("LOG_FILE", "logs/alerts.log")

# ---------------- CLASSES ----------------
NORMAL_CLASS = os.getenv("NORMAL_CLASS", "Normal")

# ---------------- SEVERITY ----------------
try:
    SEV_HIGH = float(os.getenv("SEV_HIGH", "0.95"))
    SEV_MED = float(os.getenv("SEV_MED", "0.85"))
    SEV_LOW = float(os.getenv("SEV_LOW", "0.50"))
except ValueError:
    SEV_HIGH = 0.95
    SEV_MED = 0.85
    SEV_LOW = 0.50

# ---------------- PLATFORM DETECTION ----------------
IS_JETSON = (
    platform.system() == "Linux"
    and os.path.exists("/etc/nv_tegra_release")
)

USE_TENSORRT = IS_JETSON

