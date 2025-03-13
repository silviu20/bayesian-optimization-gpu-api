# insights_router.py

"""Router for Bayesian optimization insights endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal
import pandas as pd

# Import the insights functionality
from insights import (
    get_available_explainers,
    get_available_plot_types,
    generate_shap_insight,
    generate_shap_plot,
    feature_importance_summary
)

# Create a router for insights endpoints
insights_router = APIRouter(tags=["insights"])

# Cache for storing insights between requests
_insights_cache = {}

# Pydantic models
class ShapInsightConfig(BaseModel):
    """Configuration for SHAP insight generation."""
    explainer_type: str = "KernelExplainer"
    use_comp_rep: bool = False
    force_recreate: bool = False

class ShapPlotConfig(BaseModel):
    """Configuration for SHAP plot generation."""
    plot_type: Literal["bar", "beeswarm", "force", "heatmap", "scatter", "waterfall"]
    explanation_index: Optional[int] = None
    plot_title: Optional[str] = None

class FeatureImportanceConfig(BaseModel):
    """Configuration for feature importance."""
    top_n: int = 5

# Routes
@insights_router.get("/insights/explainers", response_model=List[str])
async def list_explainers():
    """List all available SHAP explainers."""
    return get_available_explainers()

@insights_router.get("/insights/plot-types", response_model=List[str])
async def list_plot_types():
    """List all available SHAP plot types."""
    return get_available_plot_types()

@insights_router.post("/optimizations/{optimizer_id}/insights/shap")
async def create_shap_insight(
    optimizer_id: str,
    config: ShapInsightConfig = Body(...)
):
    """Generate SHAP-based insights for an optimization campaign.
    
    Args:
        optimizer_id: ID of the optimization campaign
        config: Configuration for SHAP insight generation
        
    Returns:
        Status message confirming insight generation
    """
    from main import get_optimizer  # Import here to avoid circular imports
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Generate the insight if needed
        if config.force_recreate or optimizer_id not in _insights_cache:
            insight = generate_shap_insight(
                campaign=campaign,
                explainer_type=config.explainer_type,
                use_comp_rep=config.use_comp_rep
            )
            _insights_cache[optimizer_id] = insight
        
        return {"status": "success", "message": "SHAP insight generated successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insight: {str(e)}")

@insights_router.post("/optimizations/{optimizer_id}/insights/plot")
async def create_shap_plot(
    optimizer_id: str,
    config: ShapPlotConfig = Body(...)
):
    """Generate a SHAP plot for an optimization campaign.
    
    Args:
        optimizer_id: ID of the optimization campaign
        config: Configuration for plot generation
        
    Returns:
        Base64-encoded image of the plot
    """
    from main import get_optimizer  # Import here to avoid circular imports
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Generate the insight if needed
        if optimizer_id not in _insights_cache:
            insight = generate_shap_insight(campaign=campaign)
            _insights_cache[optimizer_id] = insight
        
        # Generate the plot
        img_str = generate_shap_plot(
            insight=_insights_cache[optimizer_id],
            plot_type=config.plot_type,
            explanation_index=config.explanation_index,
            plot_title=config.plot_title
        )
        
        return {"status": "success", "image": img_str}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plot: {str(e)}")

@insights_router.post("/optimizations/{optimizer_id}/insights/feature-importance")
async def get_feature_importance(
    optimizer_id: str,
    config: FeatureImportanceConfig = Body(...)
):
    """Get feature importance for an optimization campaign based on SHAP values.
    
    Args:
        optimizer_id: ID of the optimization campaign
        config: Configuration for feature importance analysis
        
    Returns:
        Feature importance rankings
    """
    from main import get_optimizer  # Import here to avoid circular imports
    
    try:
        # Get the campaign
        campaign = get_optimizer(optimizer_id)
        
        # Generate the insight if needed
        if optimizer_id not in _insights_cache:
            insight = generate_shap_insight(campaign=campaign)
            _insights_cache[optimizer_id] = insight
        
        # Generate the feature importance summary
        summary = feature_importance_summary(
            insight=_insights_cache[optimizer_id],
            top_n=config.top_n
        )
        
        return {"status": "success", "feature_importance": summary}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate feature importance: {str(e)}")