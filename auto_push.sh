#!/bin/bash


##### 
# command to run:
# ./auto_push.sh

# -------------------------------
# Auto push latest run folder to GitHub
# -------------------------------

# Go to your repo
cd ~/RICE-N-exp || exit

# Find the latest run folder inside outputs/
latest_run=$(ls -td outputs/*/ | head -n 1)

if [ -z "$latest_run" ]; then
    echo "No run folders found in outputs/"
    exit 1
fi

echo "Latest run folder: $latest_run"

# Add everything normally
git add .

# Force-add the latest run folder (overrides .gitignore)
git add -f "$latest_run"

# Commit with timestamp
git commit -m "Auto commit: $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit"

# Push to GitHub
git push origin main

echo "✅ Latest run folder pushed: $latest_run"
