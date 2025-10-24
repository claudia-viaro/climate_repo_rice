#!/usr/bin/env python3
# launcher.py

"""
# activate environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate rice_env

# run the launcher
python ~/climate_repo_rice/launcher.py &

"""
import subprocess
import sys
import time
from pathlib import Path

# -------------------------------
# Paths
# -------------------------------
project_dir = Path.home() / "climate_repo_rice"
logfile = project_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

# -------------------------------
# Command to run
# -------------------------------
cmd = [sys.executable, "-u", str(project_dir / "scripts/trainer.py")]

print(f"🚀 Starting trainer with logging to {logfile}")
print("   You can safely disconnect. Use `tail -f {}` to monitor.".format(logfile))

# -------------------------------
# Open log file for writing
# -------------------------------
with open(logfile, "w") as log:
    # Use subprocess.Popen to run trainer in background
    # - stdout/stderr go to log
    # - unbuffered output ensures live logging
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

print(f"Trainer PID: {process.pid}")

# -------------------------------
# Optional: wait for process (keeps launcher running)
# -------------------------------
try:
    while True:
        retcode = process.poll()
        if retcode is not None:
            print(f"Trainer finished with exit code {retcode}")
            break
        time.sleep(10)
except KeyboardInterrupt:
    print("Stopping trainer...")
    process.terminate()
    process.wait()
    print("Trainer terminated")
