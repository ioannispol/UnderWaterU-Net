#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 YOUR_ONEDRIVE_SHARE_LINK"
    exit 1
fi

# The OneDrive shared link is the first argument
onedrive_link="$1"

# Convert the OneDrive share link to a direct download link
direct_link=$(echo $onedrive_link | sed 's/redir?/download?/')

# Set the name of the file to download to
output_filename="dataset.zip"

# Use wget or curl to download the file
# Uncomment the line for the tool you have available

# Using wget
wget -O "$output_filename" "$direct_link"

# Using curl
# curl -L "$direct_link" -o "$output_filename"

echo "Download complete: $output_filename"
