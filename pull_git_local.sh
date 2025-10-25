#!/bin/bash
#########################
#chmod +x pull_git_local.sh   # only once

# pull_git_local.sh
# Usage: ./pull_git_local.sh outputs/1761304985
# from github to local machine, move the output folder
#########################

cd ~/climate_repo_rice || { echo "Cannot enter project dir"; exit 1; }

run_folder="$1"
if ! git ls-tree -r origin/main --name-only | grep -q "^$run_folder/"; then
    echo "❌ $run_folder does not exist on GitHub."
    exit 1
fi

if [ -z "$run_folder" ]; then
    echo "Usage: $0 <outputs/subfolder>"
    exit 1
fi

echo "Pulling $run_folder from GitHub..."

# Fetch latest changes
git fetch origin

# Checkout the folder from remote forcefully
git checkout origin/main -- "$run_folder"

# Now make the folder untracked again (respecting .gitignore)
git rm --cached -r "$run_folder"

echo "✅ $run_folder pulled from GitHub and now untracked locally."
