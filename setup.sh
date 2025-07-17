#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit/

# Create config file
echo "\
[server]\n\nheadless = true\nport = $PORT\nenableCORS = false\n\n[browser]\nserverAddress = '0.0.0.0'\nserverPort = $PORT\n\n[theme]\nbase = 'light'\n\n" > ~/.streamlit/config.toml
