#!/bin/bash
set -e

# Check for GPU availability
echo "Checking GPU configuration..."
if [ "${USE_GPU}" = "true" ]; then
    python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'PyTorch version: {torch.__version__}')"
    
    # Run a more comprehensive GPU test
    python /app/gpu_test.py || echo "GPU test failed, but continuing startup process"
    
    # Log GPU status
    if [ -x "$(command -v nvidia-smi)" ]; then
        echo "GPU Information:"
        nvidia-smi
    else
        echo "nvidia-smi not found. GPU status cannot be displayed."
    fi
else
    echo "GPU usage disabled by configuration. Using CPU only."
fi

# Start the API service
echo "Starting Bayesian Optimization API..."
exec uvicorn main:app --host 0.0.0.0 --port 8000