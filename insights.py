#insights

"""Module for generating insights for Bayesian optimization experiments."""

from typing import Dict, Any, List, Optional, Literal
import io
import base64
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
from baybe.insights.shap import SHAPInsight, EXPLAINERS, SHAP_PLOTS
from baybe import Campaign
from fastapi import HTTPException

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_available_explainers() -> List[str]:
    """Get a list of available SHAP explainers.
    
    Returns:
        List of available explainer names
    """
    return list(EXPLAINERS)


def get_available_plot_types() -> List[str]:
    """Get a list of available SHAP plot types.
    
    Returns:
        List of available plot types
    """
    return list(SHAP_PLOTS)


def generate_shap_insight(
    campaign: Campaign,
    explainer_type: str = "KernelExplainer",
    use_comp_rep: bool = False
) -> SHAPInsight:
    """Generate a SHAP insight from a BayBE campaign.
    
    Args:
        campaign: The BayBE campaign
        explainer_type: Type of SHAP explainer to use
        use_comp_rep: Whether to use computational representation
        
    Returns:
        SHAP insight object
        
    Raises:
        HTTPException: If the campaign doesn't have measurements or other errors occur
    """
    try:
        return SHAPInsight.from_campaign(
            campaign=campaign,
            explainer_cls=explainer_type,
            use_comp_rep=use_comp_rep
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SHAP insights: {str(e)}")


def generate_shap_plot(
    insight: SHAPInsight,
    plot_type: Literal["bar", "beeswarm", "force", "heatmap", "scatter", "waterfall"],
    data: Optional[pd.DataFrame] = None,
    explanation_index: Optional[int] = None,
    plot_title: Optional[str] = None,
    **kwargs
) -> str:
    """Generate a SHAP plot and return it as a base64-encoded image.
    
    Args:
        insight: SHAP insight object
        plot_type: Type of plot to generate
        data: DataFrame to use for explanation (uses background data if None)
        explanation_index: Index of data point to explain (for single-point plots)
        plot_title: Optional title for the plot
        **kwargs: Additional arguments to pass to the plot function
        
    Returns:
        Base64-encoded image of the plot
    """
    try:
        # Create figure with title if provided
        if plot_title:
            plt.figure(figsize=(10, 6))
            plt.suptitle(plot_title)
            
        # Generate the plot
        insight.plot(
            plot_type=plot_type,
            data=data,
            explanation_index=explanation_index,
            show=False,
            **kwargs
        )
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    except Exception as e:
        plt.close()
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")


def feature_importance_summary(
    insight: SHAPInsight,
    data: Optional[pd.DataFrame] = None,
    top_n: int = 5
) -> Dict[str, Any]:
    """Generate a summary of feature importance based on SHAP values.
    
    Args:
        insight: SHAP insight object
        data: DataFrame to use for explanation (uses background data if None)
        top_n: Number of top features to include in summary
        
    Returns:
        Dictionary with feature importance information
    """
    try:
        # Get SHAP explanation
        explanation = insight.explain(data)
        
        # Calculate absolute mean SHAP values for each feature
        feature_importance = {}
        for i, feature in enumerate(explanation.feature_names):
            feature_importance[feature] = abs(explanation.values[:, i]).mean()
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create summary dict
        summary = {
            "top_features": [
                {"feature": f, "importance": float(v)} 
                for f, v in sorted_features[:top_n]
            ],
            "all_features": [
                {"feature": f, "importance": float(v)} 
                for f, v in sorted_features
            ]
        }
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating feature importance: {str(e)}")