######################### to be done after training (after running run_cluster.sh)
# chmod +x pull_git_cluster.sh   # only once
# ./pull_git_cluster.sh

cd ~/climate_repo_rice || { echo "❌ Cannot enter project dir"; exit 1; }

# Ensure repository exists
if [ ! -d ".git" ]; then
    echo "❌ Repository not found. Please clone it first."
    exit 1
fi

# Stash any local changes to tracked files (optional)
git stash push -m "auto-stash before pull"

# Pull latest changes from GitHub
git fetch origin
git reset --hard origin/main

# Optionally, pop your stash (if needed)
git stash pop || echo "No stashed changes"

echo "✅ Repository updated from GitHub, outputs/ folder preserved"
