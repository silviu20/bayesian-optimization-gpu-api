"""Test client for the BayBE insights API."""

import requests
import json
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

# API base URL
API_URL = "http://localhost:8000"  # Update with your actual API URL

def get_available_explainers():
    """Get list of available SHAP explainers."""
    response = requests.get(f"{API_URL}/insights/explainers")
    response.raise_for_status()
    return response.json()

def get_available_plot_types():
    """Get list of available SHAP plot types."""
    response = requests.get(f"{API_URL}/insights/plot-types")
    response.raise_for_status()
    return response.json()

def generate_shap_insight(optimizer_id, explainer_type="KernelExplainer", use_comp_rep=False, force_recreate=False):
    """Generate SHAP insight for an optimization."""
    payload = {
        "explainer_type": explainer_type,
        "use_comp_rep": use_comp_rep,
        "force_recreate": force_recreate
    }
    
    response = requests.post(
        f"{API_URL}/optimizations/{optimizer_id}/insights/shap", 
        json=payload
    )
    response.raise_for_status()
    return response.json()

def generate_shap_plot(optimizer_id, plot_type, explanation_index=None, plot_title=None):
    """Generate SHAP plot for an optimization."""
    payload = {
        "plot_type": plot_type,
        "explanation_index": explanation_index,
        "plot_title": plot_title
    }
    
    response = requests.post(
        f"{API_URL}/optimizations/{optimizer_id}/insights/plot", 
        json=payload
    )
    response.raise_for_status()
    
    # Get the base64-encoded image
    result = response.json()
    return result, decode_image(result["image"]) if "image" in result else None

def get_feature_importance(optimizer_id, top_n=5):
    """Get feature importance for an optimization."""
    payload = {"top_n": top_n}
    
    response = requests.post(
        f"{API_URL}/optimizations/{optimizer_id}/insights/feature-importance", 
        json=payload
    )
    response.raise_for_status()
    return response.json()

def decode_image(base64_string):
    """Decode base64 string to PIL Image."""
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

def save_image(image, filename):
    """Save PIL Image to file."""
    image.save(filename)
    print(f"Image saved to {filename}")

if __name__ == "__main__":
    # Replace with your actual optimizer ID
    optimizer_id = "example_optimizer"
    
    # List available explainers and plot types
    print("Available explainers:", get_available_explainers())
    print("Available plot types:", get_available_plot_types())
    
    # Generate SHAP insight
    print("Generating SHAP insight...")
    insight_result = generate_shap_insight(optimizer_id)
    print(insight_result)
    
    # Generate different plot types
    for plot_type in ["bar", "beeswarm", "scatter"]:
        print(f"Generating {plot_type} plot...")
        result, image = generate_shap_plot(
            optimizer_id, 
            plot_type=plot_type,
            plot_title=f"Feature Importance ({plot_type})"
        )
        if image:
            save_image(image, f"shap_{plot_type}.png")
    
    # Get feature importance
    print("Getting feature importance...")
    importance = get_feature_importance(optimizer_id)
    print(json.dumps(importance, indent=2))
    
    # Print top features
    if "feature_importance" in importance and "top_features" in importance["feature_importance"]:
        print("\nTop features by importance:")
        for i, feature in enumerate(importance["feature_importance"]["top_features"], 1):
            print(f"{i}. {feature['feature']}: {feature['importance']:.4f}")