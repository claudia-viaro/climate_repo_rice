#!/usr/bin/env python3
import subprocess
import os
import datetime
import signal
import sys

HOME = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME, "climate_repo_rice")
RAY_TMPDIR = os.path.join(HOME, "ray_tmp")

# Ensure directories exist
os.makedirs(RAY_TMPDIR, exist_ok=True)

# ----------------------------
# Stop existing Ray sessions
# ----------------------------
subprocess.run("ray stop || true", shell=True)
subprocess.run("killall -9 ray || true", shell=True)

# ----------------------------
# Clean Ray temp files
# ----------------------------
for root, dirs, files in os.walk(RAY_TMPDIR):
    for name in files:
        try:
            os.remove(os.path.join(root, name))
        except Exception:
            pass
    for name in dirs:
        try:
            os.rmdir(os.path.join(root, name))
        except Exception:
            pass

# ----------------------------
# Setup logging
# ----------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(PROJECT_DIR, f"run_{timestamp}.log")
print(f"🪵 Logging to {logfile}")

# ----------------------------
# Launch trainer
# ----------------------------
with open(logfile, "a") as logf:
    process = subprocess.Popen(
        ["python", "-u", os.path.join(PROJECT_DIR, "scripts/trainer.py")],
        stdout=logf,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # ensure it survives SSH disconnect
    )
    print(f"🚀 Trainer started with PID {process.pid}")
    print("   You can safely disconnect from SSH.")
    print(f"   Monitor progress with: tail -f {logfile}")
