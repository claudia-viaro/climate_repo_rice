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
# Stop any old Ray session
# -------------------------
if command -v ray &> /dev/null; then
    echo "🛑 Stopping any existing Ray sessions..."
    ray stop || true
fi

# -------------------------
# Clean Ray temp directory
# -------------------------
export RAY_TMPDIR=$HOME/ray_tmp
echo "🗑️ Cleaning old Ray temp files..."
rm -rf $RAY_TMPDIR/*
mkdir -p $RAY_TMPDIR
echo "🗂️ Ray temp directory: $RAY_TMPDIR"

# -------------------------
# Start Ray head node
# -------------------------
echo "⚡ Starting a fresh local Ray head node..."
ray start --head --port=6379 --temp-dir=$RAY_TMPDIR || { echo "Failed to start Ray"; exit 1; }

# -------------------------
# Run trainer in background with logging
# -------------------------
LOGFILE=~/climate_repo_rice/run_$(date +%Y%m%d_%H%M%S).log
echo "🪵 Logging to $LOGFILE"
nohup python -u scripts/trainer.py > "$LOGFILE" 2>&1 &
TRAIN_PID=$!
echo "🚀 Training started with PID $TRAIN_PID"
echo "   You can safely close the SSH session."
echo "   Monitor progress with:"
echo "      tail -f $LOGFILE"

# -------------------------
# Optional: print Ray status
# -------------------------
echo "ℹ️ Ray cluster status:"
ray status