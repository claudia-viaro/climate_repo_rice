#!/bin/bash
#SBATCH --job-name=rice_train
#SBATCH --output=rice_train_%j.log       # Slurm stdout/stderr
#SBATCH --time=3-00:00:00                # 3 days max runtime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1                     # Remove if you don't use GPU

# -------------------------
# Move to project directory
# -------------------------
cd ~/climate_repo_rice || { echo "Cannot enter project dir"; exit 1; }

# -------------------------
# Update repo from GitHub
# -------------------------
if [ -d ".git" ]; then
    echo "Repo exists, pulling latest changes..."
    git reset --hard HEAD
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
# Run the trainer
# -------------------------
LOGFILE=~/climate_repo_rice/run_$(date +%Y%m%d_%H%M%S)_slurm.log
echo "Logging to $LOGFILE"
python scripts/trainer.py > "$LOGFILE" 2>&1

# -------------------------
# Deactivate environment
# -------------------------
conda deactivate
