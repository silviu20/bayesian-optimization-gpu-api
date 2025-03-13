# initialization.py

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Body, Depends, Query
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

# Create a router for initialization endpoints
initialization_router = APIRouter(tags=["initialization"])

# Try to import pyDOE2 for Latin Hypercube Sampling
try:
    from pyDOE2 import lhs
    LHS_AVAILABLE = True
except ImportError:
    logger.warning("pyDOE2 not installed. Using numpy random sampling instead of LHS.")
    LHS_AVAILABLE = False

# Pydantic models for request/response schemas
class InitialSamplesRequest(BaseModel):
    n_samples: int = Field(10, gt=0, description="Number of initial samples to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class ExistingDataRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Existing experimental data as a dictionary of parameter_name: [values]")

class CSVUploadResponse(BaseModel):
    status: str
    message: str
    parameters: List[str]
    rows_parsed: int

# Utility functions
def generate_samples(
    parameters: List[Dict[str, Any]], 
    n_samples: int = 10,
    seed: Optional[int] = None,
    use_lhs: bool = True
) -> pd.DataFrame:
    """
    Generate initial samples using Latin Hypercube Sampling (LHS) or random sampling.
    
    Args:
        parameters: List of parameter configurations.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        use_lhs: Whether to use LHS (if available) or random sampling.
        
    Returns:
        Dataframe with parameter values for initial samples.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Extract numerical parameters and their bounds/values
    numerical_params = []
    numerical_bounds = []
    
    for param in parameters:
        if param["type"] in ["NumericalContinuous", "NumericalContinuousParameter", 
                            "NumericalDiscrete", "NumericalDiscreteParameter"]:
            numerical_params.append(param["name"])
            
            if "bounds" in param:
                # For continuous parameters with bounds
                numerical_bounds.append((param["bounds"][0], param["bounds"][1]))
            elif "values" in param:
                # For discrete parameters with values
                numerical_bounds.append((min(param["values"]), max(param["values"])))
    
    # Generate samples for numerical parameters
    n_numerical = len(numerical_params)
    df_samples = pd.DataFrame(index=range(n_samples))
    
    if n_numerical > 0:
        if use_lhs and LHS_AVAILABLE:
            # Generate normalized LHS samples in [0,1]
            lhs_samples = lhs(n_numerical, samples=n_samples)
            
            # Scale to actual parameter ranges
            for i, param_name in enumerate(numerical_params):
                lower, upper = numerical_bounds[i]
                df_samples[param_name] = lhs_samples[:,i] * (upper - lower) + lower
        else:
            # Use random sampling
            for i, param_name in enumerate(numerical_params):
                lower, upper = numerical_bounds[i]
                df_samples[param_name] = np.random.uniform(lower, upper, n_samples)
        
        # For discrete parameters, map to nearest discrete value
        for param in parameters:
            if param["type"] in ["NumericalDiscrete", "NumericalDiscreteParameter"] and "values" in param:
                name = param["name"]
                if name in df_samples.columns:
                    # Find closest discrete value for each sample
                    values = np.array(param["values"])
                    df_samples[name] = df_samples[name].apply(
                        lambda x: values[np.abs(values - x).argmin()]
                    )
    
    # Handle categorical parameters by random sampling
    for param in parameters:
        if param["type"] in ["Categorical", "CategoricalParameter", "Substance", "SubstanceParameter"]:
            if "values" in param:
                df_samples[param["name"]] = np.random.choice(
                    param["values"], 
                    size=n_samples
                )
    
    return df_samples

def parse_existing_data(
    data: Dict[str, List[Any]],
    parameters: List[Dict[str, Any]],
    target_name: str
) -> pd.DataFrame:
    """
    Parse existing experimental data provided by the user.
    
    Args:
        data: Dictionary with parameter names as keys and lists of values.
        parameters: List of parameter configurations.
        target_name: Name of the target variable.
        
    Returns:
        Dataframe with parameter values and target values.
    """
    # Create DataFrame from provided data
    df = pd.DataFrame(data)
    
    # Validate that all required parameters are present
    param_names = [p["name"] for p in parameters]
    missing_params = [p for p in param_names if p not in df.columns]
    
    if missing_params:
        raise ValueError(f"Missing parameters in provided data: {missing_params}")
    
    # Validate that target column is present
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in provided data")
    
    # Validate data types and convert if necessary
    for param in parameters:
        name = param["name"]
        
        if param["type"] in ["NumericalContinuous", "NumericalContinuousParameter",
                            "NumericalDiscrete", "NumericalDiscreteParameter"]:
            # Convert to float for numerical parameters
            df[name] = df[name].astype(float)
        elif param["type"] in ["Categorical", "CategoricalParameter", "Substance", "SubstanceParameter"]:
            # Convert to string for categorical parameters
            df[name] = df[name].astype(str)
    
    # Convert target to float
    df[target_name] = df[target_name].astype(float)
    
    return df

# Routes
@initialization_router.post("/optimizations/{optimizer_id}/initialize/predefined")
async def initialize_with_predefined_samples(
    optimizer_id: str,
    request: InitialSamplesRequest = Body(...)
):
    """
    Initialize optimization with predefined samples using Latin Hypercube Sampling.
    
    This endpoint generates initial samples using Latin Hypercube Sampling (LHS)
    based on the parameter definitions in the optimization campaign.
    """
    from main import get_optimizer  # Import here to avoid circular imports
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Get parameters from the campaign
        parameters = []
        for param in campaign.searchspace.parameters:
            # Convert BayBE parameter objects to dictionaries
            param_dict = {
                "name": param.name,
                "type": param.__class__.__name__
            }
            
            if hasattr(param, "values") and param.values is not None:
                param_dict["values"] = param.values.tolist() if hasattr(param.values, 'tolist') else param.values
            elif hasattr(param, "bounds") and param.bounds is not None:
                param_dict["bounds"] = [param.bounds.lower, param.bounds.upper]
                
            parameters.append(param_dict)
        
        # Generate samples
        samples_df = generate_samples(
            parameters=parameters,
            n_samples=request.n_samples,
            seed=request.seed,
            use_lhs=LHS_AVAILABLE
        )
        
        return {
            "status": "success",
            "message": f"Generated {len(samples_df)} initial samples",
            "samples": samples_df.to_dict(orient="records")
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating initial samples: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate initial samples: {str(e)}")

@initialization_router.post("/optimizations/{optimizer_id}/initialize/zero")
async def initialize_from_zero(
    optimizer_id: str,
    request: InitialSamplesRequest = Body(...)
):
    """
    Initialize optimization from zero samples.
    
    This endpoint generates initial samples that will be used before 
    starting the Bayesian optimization process from scratch.
    """
    # This is essentially the same as the predefined samples endpoint
    return await initialize_with_predefined_samples(optimizer_id, request)

@initialization_router.post("/optimizations/{optimizer_id}/initialize/existing")
async def initialize_with_existing_data(
    optimizer_id: str,
    request: ExistingDataRequest = Body(...)
):
    """
    Initialize optimization with existing experimental data.
    
    This endpoint allows users to provide their own experimental data to
    bootstrap the optimization process.
    """
    from main import get_optimizer, bo_backend  # Import here to avoid circular imports
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Get parameters from the campaign
        parameters = []
        for param in campaign.searchspace.parameters:
            # Convert BayBE parameter objects to dictionaries
            param_dict = {
                "name": param.name,
                "type": param.__class__.__name__
            }
            
            if hasattr(param, "values") and param.values is not None:
                param_dict["values"] = param.values
            elif hasattr(param, "bounds") and param.bounds is not None:
                param_dict["bounds"] = [param.bounds.lower, param.bounds.upper]
                
            parameters.append(param_dict)
        
        # Get target name
        target_name = campaign.targets[0].name
        
        # Parse existing data
        measurements_df = parse_existing_data(
            data=request.data,
            parameters=parameters,
            target_name=target_name
        )
        
        # Add measurements to the campaign
        result = bo_backend.add_multiple_measurements(
            optimizer_id=optimizer_id,
            measurements=measurements_df
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "message": f"Added {len(measurements_df)} measurements from existing data",
            "measurements_count": len(measurements_df)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing existing data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process existing data: {str(e)}")

@initialization_router.get("/optimizations/{optimizer_id}/parameter-info")
async def get_parameter_info(optimizer_id: str):
    """
    Get parameter information for an optimization campaign.
    
    This endpoint is useful for the front-end to create form fields for
    uploading or entering existing data.
    """
    from main import get_optimizer  # Import here to avoid circular imports
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Get parameters from the campaign
        parameters = []
        for param in campaign.searchspace.parameters:
            # Convert BayBE parameter objects to dictionaries
            param_dict = {
                "name": param.name,
                "type": param.__class__.__name__
            }
            
            if hasattr(param, "values") and param.values is not None:
                param_dict["values"] = param.values
            elif hasattr(param, "bounds") and param.bounds is not None:
                param_dict["bounds"] = [param.bounds.lower, param.bounds.upper]
                
            parameters.append(param_dict)
        
        # Get target information
        target = campaign.targets[0]
        target_info = {
            "name": target.name,
            "mode": target.mode
        }
        
        if hasattr(target, "bounds") and target.bounds is not None:
            target_info["bounds"] = [target.bounds.lower, target.bounds.upper]
        
        return {
            "status": "success",
            "parameters": parameters,
            "target": target_info
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting parameter info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get parameter info: {str(e)}")

@initialization_router.post("/optimizations/{optimizer_id}/upload-csv")
async def upload_csv_data(
    optimizer_id: str,
    file: bytes = Body(..., description="CSV file content"),
    header: bool = Query(True, description="Whether the CSV has a header row")
):
    """
    Upload a CSV file with experimental data.
    
    This endpoint allows users to upload experimental data in CSV format
    to bootstrap the optimization process.
    """
    from main import get_optimizer, bo_backend  # Import here to avoid circular imports
    import io
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Get parameter names
        param_names = [param.name for param in campaign.searchspace.parameters]
        target_name = campaign.targets[0].name
        
        # Parse CSV data
        csv_file = io.StringIO(file.decode('utf-8'))
        if header:
            df = pd.read_csv(csv_file)
        else:
            # If no header, assume columns match parameter order + target
            all_columns = param_names + [target_name]
            df = pd.read_csv(csv_file, header=None, names=all_columns)
        
        # Validate columns
        missing_params = [p for p in param_names if p not in df.columns]
        if missing_params:
            raise ValueError(f"Missing parameters in CSV: {missing_params}")
        
        if target_name not in df.columns:
            raise ValueError(f"Target column '{target_name}' not found in CSV")
        
        # Add measurements to the campaign
        result = bo_backend.add_multiple_measurements(
            optimizer_id=optimizer_id,
            measurements=df
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "message": f"Uploaded {len(df)} measurements from CSV",
            "parameters": param_names,
            "rows_parsed": len(df)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process CSV data: {str(e)}")