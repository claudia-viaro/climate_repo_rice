#!/bin/bash

# =====================================================
# ssh cluster.lip6.fr
# enter the password twice 
# cd ~/climate_repo_rice
# chmod +x run_cluster.sh (actually to be done once only)
# ./cluster_git_update.sh
# =====================================================

# Go to home directory
cd ~ || { echo "❌ Cannot access home directory"; exit 1; }

# Create parent directory if missing
mkdir -p ~/climate_repo_rice
cd ~/climate_repo_rice || { echo "❌ Cannot enter project directory"; exit 1; }

# Check if repo already exists
if [ -d ".git" ]; then
    echo "📦 Repo exists, pulling latest changes..."
    git reset --hard HEAD             # discard local changes
    echo "my_project/configs/region_yamls/" >> .git/info/exclude 2>/dev/null
    git pull origin main
else
    echo "🌱 Cloning repo fresh..."
    rm -rf ~/climate_repo_rice/*
    git clone https://github.com/claudia-viaro/climate_repo_rice.git .
fi

echo "✅ Repository up to date."
