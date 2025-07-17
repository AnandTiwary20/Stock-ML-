#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
mkdir -p ~/.streamlit/

# Create config file
echo "\
[server]\n\nheadless = true\nport = $PORT\nenableCORS = false\nmaxUploadSize = 1000\nmaxMessageSize = 1000\n\n[browser]\nserverAddress = '0.0.0.0'\nserverPort = $PORT\n\n[theme]\nbase = 'light'\n\n[logger]\nlevel = 'debug'\n\n[client]\nshowErrorDetails = true\n\n" > ~/.streamlit/config.toml

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/app"

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
