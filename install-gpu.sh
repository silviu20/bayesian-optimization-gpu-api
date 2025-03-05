#!/bin/bash
# Script to install NVIDIA Docker components on Linux
# Run as root (sudo)

set -e

echo "=== Installing NVIDIA GPU components for Docker ==="

# Verify the system has a CUDA-capable GPU
if ! command -v lspci &> /dev/null; then
    echo "Installing pciutils..."
    apt-get update && apt-get install -y pciutils
fi

if ! lspci | grep -i nvidia &> /dev/null; then
    echo "Error: No NVIDIA GPU detected. This script requires an NVIDIA GPU."
    exit 1
fi

echo "NVIDIA GPU detected."

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add current user to docker group (to avoid using sudo with docker commands)
    usermod -aG docker $SUDO_USER
    
    echo "Docker installed successfully!"
else
    echo "Docker is already installed."
fi

# Install NVIDIA drivers if not already installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Installing NVIDIA drivers..."
    apt-get update
    apt-get install -y linux-headers-$(uname -r)
    apt-get install -y nvidia-driver-525  # Adjust version as needed
    
    echo "NVIDIA drivers installed. A system reboot may be required."
else
    echo "NVIDIA drivers are already installed."
    echo "Current driver version:"
    nvidia-smi | grep "Driver Version"
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
echo "Configuring Docker to use NVIDIA runtime..."
tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Restart Docker to apply changes
systemctl restart docker

echo "Testing NVIDIA Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo "=== Installation complete! ==="
echo "You can now run GPU-accelerated Docker containers."
echo "After installation, you might need to log out and log back in for the docker group changes to take effect."