#!/bin/bash

# Navigate to your repository directory
# cd /path/to/your/repo

# Function to set git user configuration
setup_git_config() {
  echo "Configuring git user information..."
  echo "Enter your git user email:"
  read git_email
  git config --global user.email "$git_email"

  echo "Enter your git user name:"
  read git_name
  git config --global user.name "$git_name"
}

# Check if git user is set, if not set it
git_user_name=$(git config --global user.name)
git_user_email=$(git config --global user.email)

if [[ -z "$git_user_name" || -z "$git_user_email" ]]; then
  setup_git_config
fi

# Check git status
git status

# Add changes to the index
git add .

# Prompt user for a commit message
echo "Enter commit message:"
read commit_message

# Commit changes
git commit -m "$commit_message"

# Push changes to the remote repository
echo "Pushing to remote repository..."
git push

echo "Git operations completed."
