#!/bin/bash


############################## code to run on powershell
# ssh cluster.lip6.fr
# enter the password twice 
# cd ~/climate_repo_rice
# chmod +x run_cluster.sh (actually to be done once only)
# ./run_cluster.sh

# -------------------------
# Move to project directory
# -------------------------
cd ~/climate_repo_rice || { echo "Cannot enter project dir"; exit 1; }

# -------------------------
# Update repo from GitHub
# -------------------------
if [ -d ".git" ]; then
    echo "Repo exists, pulling latest changes..."
    git reset --hard HEAD         # discard any local changes
    git pull origin main
else
    echo "Cloning repo fresh..."
    git clone https://github.com/claudia-viaro/climate_repo_rice.git .
fi

# -------------------------
# Activate Python environment
# -------------------------
source ~/miniconda/etc/profile.d/conda.sh
conda activate rice_env || { echo "Failed to activate conda environment"; exit 1; }

# -------------------------
# Run the trainer detached with logging
# -------------------------
LOGFILE=~/climate_repo_rice/run_$(date +%Y%m%d_%H%M%S).log
echo "Logging to $LOGFILE"
nohup python scripts/trainer.py > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID. You can safely close the SSH session."
echo "Check log in real-time with: tail -f $LOGFILE"

# -------------------------
# Optional: wait for training to finish, then push outputs
# -------------------------
# Uncomment if you want this script to wait
# wait $TRAIN_PID

# -------------------------
# Stage and commit outputs to GitHub (outputs folder)
# -------------------------
latest_output=$(ls -dt outputs/*/ | head -1)
if [ -d "$latest_output" ]; then
    echo "Latest run folder: $latest_output"
    git add "$latest_output"
    git commit -m "Add latest cluster run: $(basename $latest_output) - $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin main
else
    echo "No outputs folder found to push"
fi

# -------------------------
# Deactivate environment
# -------------------------
conda deactivate



##### you can check at any time how it is doing
# tail -f /home/viaro/climate_repo_rice/run_20251017_090902.log