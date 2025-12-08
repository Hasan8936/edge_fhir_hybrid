# Tools & Utilities

This directory contains helpful scripts and utilities for managing the edge FHIR hybrid service on Jetson Nano.

## Scripts

### `generate_dummy_models.py`
**Purpose:** Create test/dummy model artifacts for quick evaluation

**Usage:**
```bash
python3 generate_dummy_models.py
```

**Output:** Creates the 5 required model files in `models/` directory with dummy implementations that satisfy the interface but return random predictions.

**When to use:**
- Testing the service before you have trained models
- Quick prototyping
- Development and debugging

**When NOT to use:**
- Production deployments (use real trained models)
- Accuracy matters (results are random)

---

### `smoke_test.py`
**Purpose:** Verify the system can load models and run inference

**Usage:**
```bash
python3 tools/smoke_test.py
```

**Checks:**
1. Model artifacts can be loaded
2. Feature extraction works
3. Inference runs successfully
4. Output has expected format

**Output Example:**
```
============================================================
  SMOKE TEST: Model Loading and Inference
============================================================

[1/3] Testing model loading...
    ✓ Model loaded successfully

[2/3] Testing feature extraction...
    ✓ Extracted 8 features
    ✓ Metadata: {...}

[3/3] Testing inference...
    ✓ Prediction: Normal
    ✓ Score: 0.123
    ✓ Severity: LOW
    ✓ Anomaly: False

============================================================
✅ SMOKE TEST PASSED
============================================================
```

**When to use:**
- Before deploying to production
- After adding new model artifacts
- Troubleshooting model loading issues

---

### `jetson_preflight_check.sh`
**Purpose:** Automated pre-deployment verification

**Usage:**
```bash
bash tools/jetson_preflight_check.sh
```

**Checks:**
- OS and JetPack version
- Docker installation and daemon
- User permissions
- Repository structure
- Model artifact presence
- System resources (RAM, disk, swap)
- Network connectivity
- Python and dependencies

**Output Example:**
```
==========================================
  JETSON NANO PRE-FLIGHT CHECKLIST
==========================================

[1/9] Checking OS and JetPack...
    ✓ Running on Jetson OS
    ✓ Ubuntu OS detected

[2/9] Checking Docker installation...
    ✓ Docker installed: Docker version 20.10.x
    ✓ Docker Compose installed: docker-compose version 1.29.x

...

==========================================
  RESULTS
==========================================
✓ Checks Passed: 20
✗ Checks Failed: 0
⚠ Warnings: 0

✅ System is ready! You can proceed with:
   docker-compose build
   docker-compose up -d
```

**When to use:**
- Before first deployment
- After system updates
- Before going to production
- Troubleshooting startup issues

---

## Quick Command Reference

```bash
# Test if models can load
python3 tools/smoke_test.py

# Check system is ready
bash tools/jetson_preflight_check.sh

# Generate dummy models for testing
python3 generate_dummy_models.py

# View all tests/utilities
ls -la tools/
```

---

## Exit Codes

### `smoke_test.py`
- `0` - Success
- `1` - Failure (check output for details)

### `jetson_preflight_check.sh`
- `0` - All checks passed, ready to proceed
- `1` - Some checks failed, fix issues before proceeding

---

## Troubleshooting

### Script won't run
```bash
# Give execute permissions
chmod +x tools/jetson_preflight_check.sh
```

### Python script fails to import packages
```bash
# Install missing dependencies
pip3 install -r requirements.txt
```

### Permission denied on model generation
```bash
# Ensure write access to models directory
chmod 755 models/
```

---

## Adding New Tools

When adding a new utility script:

1. Place in `tools/` directory
2. Add a header comment explaining purpose
3. Document usage with examples
4. Include error handling and clear output
5. Add entry to this README
6. Test on actual Jetson hardware

---

## Maintenance

- **Keep scripts updated** as model format or requirements change
- **Test regularly** on target Jetson hardware
- **Document changes** in commit messages
- **Remove debug output** before committing to main branch

---

## Support

For issues with these tools, check:
- `JETSON_NANO_SETUP.md` - Full setup guide
- `DEPLOYMENT.md` - Deployment troubleshooting
- GitHub issues: https://github.com/Hasan8936/edge_fhir_hybrid/issues
