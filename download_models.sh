#!/bin/bash

# Exit on any error
set -e

# Function to download and extract files
download_and_extract() {
    local url=$1
    local file=$(basename "$url")
    
    echo "Downloading $file..."
    if ! wget --quiet --show-progress "$url"; then
        echo "Error: Failed to download $file"
        exit 1
    fi
    
    echo "Extracting $file..."
    if ! tar -xvzf "$file"; then
        echo "Error: Failed to extract $file"
        exit 1
    fi
    
    echo "Removing $file..."
    if ! rm "$file"; then
        echo "Warning: Failed to remove $file"
    fi
}

# Create ScaLR directory if it doesn't exist
if [ ! -d "ScaLR" ]; then
    echo "Creating ScaLR directory..."
    if ! mkdir -p ScaLR; then
        echo "Error: Failed to create ScaLR directory"
        exit 1
    fi
fi

# Change to ScaLR directory
echo "Changing to ScaLR directory..."
if ! cd ScaLR; then
    echo "Error: Failed to change to ScaLR directory"
    exit 1
fi

# Download and extract models
download_and_extract "https://github.com/valeoai/ScaLR/releases/download/v0.1.0/info_datasets.tar.gz"
download_and_extract "https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz"
download_and_extract "https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-kitti.tar.gz"
download_and_extract "https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-no_pretraining-finetuning-nuscenes-100p.tar.gz"

cd ..
