#!/bin/bash


##### 
#chmod +x push_local_git.sh   # only once

# command to run:
# ./push_local_git.sh

# -------------------------------
# Auto push latest run folder to GitHub
# -------------------------------

##### # command to run: 
# ./push_local_git.sh 


# Go to your repo 

cd ~/RICE-N-exp || exit 
echo "outputs/" >> .gitignore # won't hurt if already there 

# Add all changes 
git add . ':!outputs/*' 
# Commit with timestamp 

git commit -m "Auto commit: $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit" 
# Push to GitHub 
git push origin main