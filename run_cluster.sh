#!/bin/bash


############################## code to run on powershell
# ssh cluster.lip6.fr
# enter the password twice 
# cd ~/climate_repo_rice
# chmod +x run_cluster.sh (actually to be done once only)
# ./run_cluster.sh

# =====================================================
# run_trainer.sh — Run RICE-N training on the cluster
# =====================================================

# -------------------------
# Move to project directory
# -------------------------
PROJECT_DIR=~/climate_repo_rice
cd "$PROJECT_DIR" || { echo "❌ Cannot enter project directory: $PROJECT_DIR"; exit 1; }

# -------------------------
# Check if repo exists and show last update time
# -------------------------
if [ ! -d ".git" ]; then
    echo "❌ Error: Git repository not found in $PROJECT_DIR"
    echo "   Please run ./update_repo.sh first."
    exit 1
fi

echo "✅ Repository found in $PROJECT_DIR"

# Last update info
LAST_UPDATE=$(git log -1 --format="%cd" --date=iso-local 2>/dev/null)
LAST_COMMIT=$(git log -1 --format="%h - %s" 2>/dev/null)
echo "📅 Last update: $LAST_UPDATE"
echo "🪶 Last commit: $LAST_COMMIT"
echo "-------------------------------------------"

# -------------------------
# Activate Python environment
# -------------------------
source ~/miniconda/etc/profile.d/conda.sh
conda activate rice_env || { echo "❌ Failed to activate conda environment"; exit 1; }

# -------------------------
# Run the trainer detached with logging
# -------------------------
LOGFILE="$PROJECT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "🪵 Logging to $LOGFILE"

nohup python scripts/trainer.py > "$LOGFILE" 2>&1 &
TRAIN_PID=$!

sleep 1
if ps -p $TRAIN_PID > /dev/null; then
    echo "🚀 Training started successfully with PID $TRAIN_PID"
    echo "   You can safely close the SSH session."
    echo "   Monitor progress with:"
    echo "      tail -f $LOGFILE"
else
    echo "❌ Training failed to start. Check the log:"
    echo "   cat $LOGFILE"
fi

# -------------------------
# Deactivate environment
# -------------------------
conda deactivate