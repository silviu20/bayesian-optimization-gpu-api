import os
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd
import torch

# Import the updated GPU-enabled BayesianOptimizationBackend class
from backend import BayesianOptimizationBackend

# Load environment variables
load_dotenv()

# Get API key and GPU configuration from environment
API_KEY = os.getenv("API_KEY", "your-default-api-key")
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# Log GPU information on startup
if USE_GPU:
    if torch.cuda.is_available():
        print(f"GPU ENABLED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("WARNING: GPU requested but not available. Using CPU instead.")
else:
    print("GPU DISABLED: Using CPU for computations")

app = FastAPI(title="Bayesian Optimization API (GPU-enabled)")
backend = BayesianOptimizationBackend(storage_path="/data", use_gpu=USE_GPU)

# API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
    )

# Pydantic models
class OptimizationConfig(BaseModel):
    parameters: List[Dict[str, Any]]
    target_config: Dict[str, Any]
    recommender_config: Optional[Dict[str, Any]] = None

class MeasurementInput(BaseModel):
    parameters: Dict[str, Any]
    target_value: float

class MultipleInput(BaseModel):
    measurements: List[Dict[str, Any]]

# Endpoints with API key protection
@app.post("/optimization/{optimizer_id}")
def create_optimization(optimizer_id: str, config: OptimizationConfig, api_key: APIKey = Depends(get_api_key)):
    """Create a new optimization process."""
    result = backend.create_optimization(
        optimizer_id=optimizer_id,
        parameters=config.parameters,
        target_config=config.target_config,
        recommender_config=config.recommender_config
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.get("/optimization/{optimizer_id}/suggest")
def suggest_next_point(optimizer_id: str, batch_size: int = 1, api_key: APIKey = Depends(get_api_key)):
    """Get the next suggested point(s) to evaluate."""
    result = backend.suggest_next_point(optimizer_id, batch_size)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.post("/optimization/{optimizer_id}/measurement")
def add_measurement(optimizer_id: str, measurement: MeasurementInput, api_key: APIKey = Depends(get_api_key)):
    """Add a new measurement to the optimizer."""
    result = backend.add_measurement(
        optimizer_id=optimizer_id,
        parameters=measurement.parameters,
        target_value=measurement.target_value
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.post("/optimization/{optimizer_id}/measurements")
def add_multiple_measurements(optimizer_id: str, input_data: MultipleInput, api_key: APIKey = Depends(get_api_key)):
    """Add multiple measurements at once."""
    # Convert to dataframe
    df = pd.DataFrame(input_data.measurements)
    
    result = backend.add_multiple_measurements(
        optimizer_id=optimizer_id,
        measurements=df
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.get("/optimization/{optimizer_id}/best")
def get_best_point(optimizer_id: str, api_key: APIKey = Depends(get_api_key)):
    """Get the current best point."""
    result = backend.get_best_point(optimizer_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.get("/optimization/{optimizer_id}/load")
def load_optimization(optimizer_id: str, api_key: APIKey = Depends(get_api_key)):
    """Load an existing optimization."""
    result = backend.load_campaign(optimizer_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

# Add a health check endpoint (no auth required)
@app.get("/health")
def health_check():
    """Check API health status and GPU information."""
    gpu_info = None
    if USE_GPU and torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1e6,
            "memory_reserved_mb": torch.cuda.memory_reserved(0) / 1e6,
            "device_count": torch.cuda.device_count()
        }
    
    return {
        "status": "healthy", 
        "using_gpu": USE_GPU and torch.cuda.is_available(),
        "gpu_info": gpu_info
    }
# added new test here 02/04/2025
@app.get("/test-optimization")
def test_optimization():
    """Test endpoint to check if the optimization process works."""
    # Define some default or dummy parameters for testing
    default_config = OptimizationConfig(
        parameters={"param1": "value1"},
        target_config={"target": "value"},
        recommender_config={"recommender": "default"}
    )

    # Call the backend optimization function with a default optimizer ID
    result = backend.create_optimization(
        optimizer_id="default_optimizer",
        parameters=default_config.parameters,
        target_config=default_config.target_config,
        recommender_config=default_config.recommender_config
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return {"message": "Optimization test successful", "result": result}
    
# Document API with Swagger UI
@app.get("/")
def redirect_to_docs():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
