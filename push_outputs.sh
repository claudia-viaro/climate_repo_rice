#!/bin/bash
######################### to be done after training (after running run_cluster.sh)
#chmod +x push_outputs.sh   # only once
#./push_outputs.sh

cd ~/climate_repo_rice || { echo "Cannot enter project dir"; exit 1; }

# Activate environment if needed (optional)
source ~/miniconda/etc/profile.d/conda.sh
conda activate rice_env || echo "Warning: could not activate environment"

# -------------------------
# Detect the latest created folder in outputs
# -------------------------
latest_output=$(ls -dt outputs/*/ | head -1)

if [ -z "$latest_output" ]; then
    echo "No output folder found. Nothing to push."
    exit 1
fi

echo "Latest run folder: $latest_output"

# -------------------------
# Add and commit only the latest run folder
# -------------------------
git add "$latest_output"
git commit -m "Add latest cluster run: $(basename $latest_output) - $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main

echo "Outputs pushed to GitHub successfully."

# Optional: deactivate environment
conda deactivate
