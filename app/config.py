"""Runtime configuration and constants for the edge inference service.

Values may be overridden by environment variables to allow easy tuning
when running in Docker or on-device.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Directories inside container where model artifacts and logs are mounted
MODELS_DIR: str = os.getenv("MODELS_DIR", "/opt/app/models")
LOG_DIR: str = os.getenv("LOG_DIR", "/opt/app/logs")
LOG_FILE: str = os.getenv("LOG_FILE", f"{LOG_DIR}/alerts.log")

# Normal class label expected from label encoder
NORMAL_CLASS: str = os.getenv("NORMAL_CLASS", "Normal")

# Severity thresholds (can be overridden via env vars)
try:
	SEV_HIGH: float = float(os.getenv("SEV_HIGH", "0.95"))
except ValueError:
	SEV_HIGH = 0.95
try:
	SEV_MED: float = float(os.getenv("SEV_MED", "0.85"))
except ValueError:
	SEV_MED = 0.85

def ensure_paths_exist(log_dir: Optional[str] = None, models_dir: Optional[str] = None) -> None:
	"""Ensure log and models directories exist (no-op if they already do).

	Parameters
	----------
	log_dir: Optional[str]
		Directory for logs. If None, uses configured `LOG_DIR`.
	models_dir: Optional[str]
		Directory for models. If None, uses configured `MODELS_DIR`.
	"""
	Path(log_dir or LOG_DIR).mkdir(parents=True, exist_ok=True)
	Path(models_dir or MODELS_DIR).mkdir(parents=True, exist_ok=True)

