#!/bin/bash
# Script to make all necessary files executable

# Make scripts executable
chmod +x entrypoint.sh
chmod +x launcher.sh
chmod +x setup.sh
chmod +x install-gpu.sh
chmod +x gpu_test.py
chmod +x profile-performance.py
chmod +x test-client.py

echo "All scripts are now executable."