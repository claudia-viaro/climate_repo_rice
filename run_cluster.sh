#!/bin/bash


############################## code to run on powershell
# ssh cluster.lip6.fr
# enter the password twice 
# cd ~/climate_repo_rice
# chmod +x run_cluster.sh (actually to be done once only)
# run_cluster.sh

# 1. Go to home directory (or wherever you keep repos)
cd ~/climate_repo_rice || exit
# 2. Update repo
if [ -d ".git" ]; then
    echo "Repo exists, pulling latest changes..."
    git reset --hard HEAD
    git pull origin main
else
    echo "Cloning repo fresh..."
    git clone https://github.com/claudia-viaro/climate_repo_rice.git .
fi

# 3. Activate your Python environment
# Adjust to your setup: either venv or Miniconda
# For venv:
source ~/miniconda3/etc/profile.d/conda.sh   # adjust path if needed
conda activate rice_env
# 4. Run the trainer
python scripts/trainer.py

# 5. Stage and commit outputs to GitHub
# Detect the latest created folder in outputs
latest_output=$(ls -dt outputs/*/ | head -1)

echo "Latest run folder: $latest_output"

# Add and commit only the latest run folder
git add "$latest_output"
git commit -m "Add latest cluster run: $(basename $latest_output) - $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main

# Deactivate environment
conda deactivate