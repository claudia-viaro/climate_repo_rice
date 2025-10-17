#!/bin/bash


############################## code to run on powershell
# ssh cluster.lip6.fr
# enter the password twice 
# cd ~/climate_repo_rice
# chmod +x run_cluster.sh (actually to be done once only)
# run_cluster.sh

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
# Adjust path: your Miniconda is in ~/miniconda (parent directory)
source ~/miniconda/etc/profile.d/conda.sh
conda activate rice_env || { echo "Failed to activate conda environment"; exit 1; }

# -------------------------
# Run the trainer detached with logging
# -------------------------
LOGFILE=~/climate_repo_rice/run_$(date +%Y%m%d_%H%M%S).log
echo "Logging to $LOGFILE"
nohup python scripts/trainer.py > "$LOGFILE" 2>&1 &

echo "Training started. You can safely close the SSH session."
echo "Check log in real-time with: tail -f $LOGFILE"

