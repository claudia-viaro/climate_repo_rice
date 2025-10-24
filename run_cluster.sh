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
# Check Git repo
# -------------------------
if [ -d ".git" ]; then
    echo "✅ Repository found in $(pwd)"
    echo -n "📅 Last update: "
    git log -1 --format=%cd
    echo -n "🪶 Last commit: "
    git log -1 --format="%h - %s"
else
    echo "❌ Repository not found. Please run cluster_git_update.sh first."
    exit 1
fi

# -------------------------
# Activate Python environment
# -------------------------
source ~/miniconda/etc/profile.d/conda.sh
conda activate rice_env || { echo "Failed to activate conda environment"; exit 1; }

# -------------------------
# Set up Ray temp directory
# -------------------------
export RAY_TMPDIR=$HOME/ray_tmp
mkdir -p $RAY_TMPDIR
echo "🗂️ Using Ray temp directory: $RAY_TMPDIR"

# -------------------------
# Start Ray head node
# -------------------------
ray start --head --temp-dir=$RAY_TMPDIR --object-store-memory 1000000000 || {
    echo "❌ Failed to start Ray head node"; exit 1;
}
echo "✅ Ray head node started"

# -------------------------
# Run trainer detached with logging
# -------------------------
LOGFILE=~/climate_repo_rice/run_$(date +%Y%m%d_%H%M%S).log
echo "🪵 Logging to $LOGFILE"
nohup python scripts/trainer.py > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "🚀 Training started with PID $TRAIN_PID"
echo "   You can safely close the SSH session."
echo "   Monitor progress with:"
echo "      tail -f $LOGFILE"

# -------------------------
# Optional: wait for training
# -------------------------
# wait $TRAIN_PID

# -------------------------
# Deactivate environment
# -------------------------
conda deactivate