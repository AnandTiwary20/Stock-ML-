#!/bin/bash
set -e  # Exit on error

# Install system dependencies
echo "Installing system dependencies..."
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

# Install Python dependencies with specific versions to avoid conflicts
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing core dependencies..."
pip install \
    numpy==1.21.6 \
    pandas==1.3.5 \
    matplotlib==3.5.3

echo "Installing data fetching dependencies..."
pip install \
    yfinance==0.2.3 \
    requests==2.28.1 \
    pandas-datareader==0.10.0

echo "Installing machine learning dependencies..."
pip install \
    "protobuf>=3.9.2,<3.20" \
    "tensorflow-cpu==2.9.1" \
    "scikit-learn==1.0.2"

echo "Installing Streamlit..."
pip install "streamlit==1.12.0"

echo "Verifying installations..."
pip freeze
