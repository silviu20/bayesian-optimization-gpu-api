# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import torch
import os
import json
import pandas as pd
from pathlib import Path

# Import the Bayesian Optimization backend
# Update the import to match your actual module name
from backend import BayesianOptimizationBackend

# Import the insights module
from insights_router import insights_router

# Initialize the FastAPI app
app = FastAPI(
    title="BayBE API", 
    description="Bayesian Optimization Backend API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(insights_router)

# Create a global instance of the BayesianOptimizationBackend
storage_path = os.environ.get("STORAGE_PATH", "./data")
use_gpu = os.environ.get("USE_GPU", "True").lower() == "true"
bo_backend = BayesianOptimizationBackend(storage_path=storage_path, use_gpu=use_gpu)

# Pydantic models for request/response schemas
class Parameter(BaseModel):
    name: str
    type: str
    values: Optional[List[Union[int, float, str]]] = None
    bounds: Optional[List[float]] = None
    encoding: Optional[str] = None
    tolerance: Optional[float] = None

class TargetConfig(BaseModel):
    name: str
    mode: str = "MAX"
    bounds: Optional[Dict[str, float]] = None
    type: Optional[str] = None
    weight: Optional[float] = None

class Condition(BaseModel):
    type: str
    threshold: Optional[float] = None
    parameter: Optional[str] = None
    values: Optional[List[Any]] = None

class Constraint(BaseModel):
    type: str
    parameters: Optional[List[str]] = None
    conditions: Optional[List[Condition]] = None
    condition: Optional[Condition] = None
    weight: Optional[float] = None
    constraint_func: Optional[str] = None
    description: Optional[str] = None

class SurrogateConfig(BaseModel):
    type: str
    kernel: Optional[Dict[str, Any]] = None
    normalize_targets: Optional[bool] = None

class AcquisitionConfig(BaseModel):
    type: str
    beta: Optional[float] = None

class RecommenderConfig(BaseModel):
    type: str
    initial_recommender: Optional[Dict[str, Any]] = None
    recommender: Optional[Dict[str, Any]] = None
    n_restarts: Optional[int] = None
    n_raw_samples: Optional[int] = None
    switch_after: Optional[int] = None
    remain_switched: Optional[bool] = None

class OptimizationConfig(BaseModel):
    parameters: List[Parameter]
    target_config: Union[TargetConfig, List[TargetConfig]]
    recommender_config: Optional[RecommenderConfig] = None
    constraints: Optional[List[Constraint]] = None
    objective_type: Optional[str] = "SingleTarget"
    surrogate_config: Optional[SurrogateConfig] = None
    acquisition_config: Optional[AcquisitionConfig] = None

class Measurement(BaseModel):
    parameters: Dict[str, Any]
    target_value: float

class Measurements(BaseModel):
    measurements: List[Dict[str, Any]]

# Helper function to get optimizer or raise an exception if not found
def get_optimizer(optimizer_id: str):
    if optimizer_id in bo_backend.active_campaigns:
        return bo_backend.active_campaigns[optimizer_id]
    
    # Try to load the optimizer
    result = bo_backend.load_campaign(optimizer_id)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=f"Optimizer not found: {optimizer_id}")
    
    return bo_backend.active_campaigns[optimizer_id]

# Routes
@app.get("/health")
async def health_check():
    """
    Check the health status of the BayBE API and GPU availability
    """
    gpu_info = None
    using_gpu = bo_backend.device.type == "cuda"
    
    if using_gpu:
        # Get GPU info
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1e6,
            "memory_cached_mb": torch.cuda.memory_cached(0) / 1e6
        }
    
    return {
        "status": "ok",
        "using_gpu": using_gpu,
        "gpu_info": gpu_info
    }

@app.post("/optimizations/{optimizer_id}/create")
async def create_optimization(optimizer_id: str, config: OptimizationConfig):
    """
    Create a new optimization process
    """
    # Convert Pydantic model to dictionary
    config_dict = config.dict(exclude_unset=True)
    
    result = bo_backend.create_optimization(
        optimizer_id=optimizer_id,
        parameters=config_dict["parameters"],
        target_config=config_dict["target_config"],
        constraints=config_dict.get("constraints"),
        recommender_config=config_dict.get("recommender_config"),
        objective_type=config_dict.get("objective_type", "SingleTarget"),
        surrogate_config=config_dict.get("surrogate_config"),
        acquisition_config=config_dict.get("acquisition_config")
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.get("/optimizations/{optimizer_id}/suggest")
async def suggest_next_point(optimizer_id: str, batch_size: int = 1):
    """
    Get suggested next point(s) to evaluate
    """
    # Make sure the optimizer exists
    get_optimizer(optimizer_id)
    
    result = bo_backend.suggest_next_point(
        optimizer_id=optimizer_id,
        batch_size=batch_size
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/optimizations/{optimizer_id}/measurement")
async def add_measurement(optimizer_id: str, measurement: Measurement):
    """
    Add a new measurement to the optimizer
    """
    # Make sure the optimizer exists
    get_optimizer(optimizer_id)
    
    result = bo_backend.add_measurement(
        optimizer_id=optimizer_id,
        parameters=measurement.parameters,
        target_value=measurement.target_value
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/optimizations/{optimizer_id}/measurements")
async def add_multiple_measurements(optimizer_id: str, data: Measurements):
    """
    Add multiple measurements at once
    """
    # Make sure the optimizer exists
    get_optimizer(optimizer_id)
    
    # Create a DataFrame from the measurements
    measurements_df = pd.DataFrame(data.measurements)
    
    result = bo_backend.add_multiple_measurements(
        optimizer_id=optimizer_id,
        measurements=measurements_df
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.get("/optimizations/{optimizer_id}/best")
async def get_best_point(optimizer_id: str):
    """
    Get the current best point
    """
    # Make sure the optimizer exists
    get_optimizer(optimizer_id)
    
    result = bo_backend.get_best_point(optimizer_id=optimizer_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.post("/optimizations/{optimizer_id}/load")
async def load_optimization(optimizer_id: str):
    """
    Load an existing optimization from disk
    """
    result = bo_backend.load_campaign(optimizer_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.get("/optimizations/{optimizer_id}/history")
async def get_measurement_history(optimizer_id: str):
    """
    Get the measurement history for an optimization
    """
    # Make sure the optimizer exists
    get_optimizer(optimizer_id)
    
    result = bo_backend.get_measurement_history(optimizer_id=optimizer_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.get("/optimizations/{optimizer_id}/info")
async def get_campaign_info(optimizer_id: str):
    """
    Get information about a campaign
    """
    # Make sure the optimizer exists
    get_optimizer(optimizer_id)
    
    result = bo_backend.get_campaign_info(optimizer_id=optimizer_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@app.get("/optimizations")
async def list_optimizations():
    """
    List all available optimizations
    """
    result = bo_backend.list_optimizations()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@app.delete("/optimizations/{optimizer_id}")
async def delete_optimization(optimizer_id: str):
    """
    Delete an optimization from memory (not from disk)
    """
    if optimizer_id in bo_backend.active_campaigns:
        del bo_backend.active_campaigns[optimizer_id]
        return {"status": "success", "message": f"Optimization {optimizer_id} removed from memory"}
    else:
        return {"status": "warning", "message": f"Optimization {optimizer_id} was not in memory"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)