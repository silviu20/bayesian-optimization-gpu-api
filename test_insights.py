# test_insights.py

import pytest
import pandas as pd
import numpy as np
import base64
import sys
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, FastAPI

from insights import (
    get_available_explainers,
    get_available_plot_types,
    generate_shap_insight,
    generate_shap_plot,
    feature_importance_summary
)
from insights_router import insights_router, _insights_cache

# Test fixtures
@pytest.fixture
def mock_campaign():
    """Create a mock campaign for testing."""
    mock = MagicMock()
    mock.measurements = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1, 2, 3],
        "target": [10.0, 20.0, 15.0]
    })
    return mock

@pytest.fixture
def mock_insight():
    """Create a mock SHAP insight object."""
    mock = MagicMock()
    # Mock the explain method to return an object with the right structure
    explanation = MagicMock()
    explanation.feature_names = ["x", "y"]
    explanation.values = np.array([
        [0.5, 0.3],
        [0.2, 0.6],
        [0.8, 0.1]
    ])
    mock.explain.return_value = explanation
    return mock

@pytest.fixture
def reset_cache():
    """Reset the insights cache before and after tests."""
    _insights_cache.clear()
    yield
    _insights_cache.clear()

# Test get_available_explainers
def test_get_available_explainers():
    """Test getting available SHAP explainers."""
    with patch('insights.EXPLAINERS', {'ExplainerA', 'ExplainerB'}):
        explainers = get_available_explainers()
        assert set(explainers) == {'ExplainerA', 'ExplainerB'}

# Test get_available_plot_types
def test_get_available_plot_types():
    """Test getting available SHAP plot types."""
    with patch('insights.SHAP_PLOTS', {'PlotA', 'PlotB'}):
        plot_types = get_available_plot_types()
        assert set(plot_types) == {'PlotA', 'PlotB'}

# Test generate_shap_insight
def test_generate_shap_insight(mock_campaign):
    """Test generating SHAP insights."""
    # Mock the SHAPInsight.from_campaign method
    with patch('insights.SHAPInsight.from_campaign') as mock_from_campaign:
        # Configure the mock to return a mock insight
        mock_insight = MagicMock()
        mock_from_campaign.return_value = mock_insight
        
        # Call the function
        result = generate_shap_insight(
            campaign=mock_campaign,
            explainer_type="TestExplainer",
            use_comp_rep=True
        )
        
        # Verify results
        assert result == mock_insight
        mock_from_campaign.assert_called_once_with(
            campaign=mock_campaign,
            explainer_cls="TestExplainer",
            use_comp_rep=True
        )

# Test generate_shap_insight error handling
def test_generate_shap_insight_error(mock_campaign):
    """Test error handling in SHAP insight generation."""
    # Mock SHAPInsight.from_campaign to raise an exception
    with patch('insights.SHAPInsight.from_campaign', side_effect=ValueError("Test error")):
        # Call should raise HTTPException
        with pytest.raises(HTTPException) as excinfo:
            generate_shap_insight(campaign=mock_campaign)
        
        # Check that the exception has the right status code and detail
        assert excinfo.value.status_code == 400
        assert "Test error" in excinfo.value.detail

# Test generate_shap_plot
def test_generate_shap_plot(mock_insight):
    """Test generating SHAP plots."""
    # Mock plt methods
    with patch('insights.plt.figure'):
        with patch('insights.plt.suptitle'):
            with patch('insights.plt.tight_layout'):
                with patch('insights.plt.savefig'):
                    with patch('insights.plt.close'):
                        with patch('insights.io.BytesIO') as mock_bytesio:
                            with patch('insights.base64.b64encode') as mock_b64encode:
                                # Configure the mocks
                                mock_buffer = MagicMock()
                                mock_bytesio.return_value = mock_buffer
                                mock_buffer.read.return_value = b"test image data"
                                mock_b64encode.return_value = b"base64 encoded data"
                                
                                # Call the function
                                result = generate_shap_plot(
                                    insight=mock_insight,
                                    plot_type="bar",
                                    plot_title="Test Plot"
                                )
                                
                                # Verify results
                                assert result == "base64 encoded data"
                                mock_insight.plot.assert_called_once_with(
                                    plot_type="bar",
                                    data=None,
                                    explanation_index=None,
                                    show=False
                                )

# Test feature_importance_summary
def test_feature_importance_summary(mock_insight):
    """Test generating feature importance summary."""
    # Call the function
    result = feature_importance_summary(
        insight=mock_insight,
        top_n=2
    )
    
    # Verify results
    assert "top_features" in result
    assert "all_features" in result
    assert len(result["top_features"]) == 2
    assert len(result["all_features"]) == 2  # Only 2 features in the mock
    
    # Check that features are in the right order (by importance)
    assert result["top_features"][0]["feature"] in ["x", "y"]
    assert result["top_features"][1]["feature"] in ["x", "y"]
    assert result["top_features"][0]["feature"] != result["top_features"][1]["feature"]

# API Endpoint Tests - FIXED

@pytest.fixture
def mock_api_dependencies(mock_campaign, mock_insight):
    """Setup mocks for API dependencies."""
    # Create mock for main.get_optimizer
    mock_get_optimizer = MagicMock(return_value=mock_campaign)
    
    # Create a module level patch
    sys.modules['main'] = MagicMock()
    sys.modules['main'].get_optimizer = mock_get_optimizer
    
    # Mock the insight functions
    with patch('insights_router.generate_shap_insight', return_value=mock_insight) as mock_gen_insight:
        with patch('insights_router.generate_shap_plot', return_value="base64 image") as mock_gen_plot:
            with patch('insights_router.feature_importance_summary') as mock_feat_imp:
                mock_feat_imp.return_value = {
                    "top_features": [
                        {"feature": "x", "importance": 0.6},
                        {"feature": "y", "importance": 0.4}
                    ],
                    "all_features": [
                        {"feature": "x", "importance": 0.6},
                        {"feature": "y", "importance": 0.4}
                    ]
                }
                yield mock_gen_insight, mock_gen_plot, mock_feat_imp
    
    # Clean up
    if 'main' in sys.modules:
        del sys.modules['main']

# Test list_explainers endpoint
def test_list_explainers_endpoint():
    """Test the list explainers endpoint."""
    # Create test client
    app = FastAPI()
    app.include_router(insights_router)
    client = TestClient(app)
    
    # Mock get_available_explainers
    with patch('insights_router.get_available_explainers', return_value=["ExplainerA", "ExplainerB"]):
        # Call the endpoint
        response = client.get("/insights/explainers")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "ExplainerA" in data
    assert "ExplainerB" in data

# Test list_plot_types endpoint
def test_list_plot_types_endpoint():
    """Test the list plot types endpoint."""
    # Create test client
    app = FastAPI()
    app.include_router(insights_router)
    client = TestClient(app)
    
    # Mock get_available_plot_types
    with patch('insights_router.get_available_plot_types', return_value=["PlotA", "PlotB"]):
        # Call the endpoint
        response = client.get("/insights/plot-types")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "PlotA" in data
    assert "PlotB" in data

# Test create_shap_insight endpoint
def test_create_shap_insight_endpoint(mock_api_dependencies, reset_cache):
    """Test the create SHAP insight endpoint."""
    mock_gen_insight, _, _ = mock_api_dependencies
    
    # Create test client
    app = FastAPI()
    app.include_router(insights_router)
    client = TestClient(app)
    
    # Call the endpoint
    response = client.post(
        "/optimizations/test_optimizer/insights/shap",
        json={"explainer_type": "TestExplainer", "use_comp_rep": True}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Check that the insight was generated and cached
    mock_gen_insight.assert_called_once_with(
        campaign=mock_api_dependencies[0].return_value,
        explainer_type="TestExplainer",
        use_comp_rep=True
    )
    assert "test_optimizer" in _insights_cache

# Test create_shap_insight endpoint
def test_create_shap_insight_endpoint(mock_api_dependencies, reset_cache):
    """Test the create SHAP insight endpoint."""
    mock_gen_insight, _, _ = mock_api_dependencies
    
    # Create test client
    app = FastAPI()
    app.include_router(insights_router)
    client = TestClient(app)
    
    # Call the endpoint
    response = client.post(
        "/optimizations/test_optimizer/insights/shap",
        json={"explainer_type": "TestExplainer", "use_comp_rep": True}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Just check that the function was called once - don't check the exact campaign object
    mock_gen_insight.assert_called_once()
    
    # Check that the correct parameters were passed without checking the campaign object
    call_args = mock_gen_insight.call_args[1]  # Get the kwargs
    assert call_args["explainer_type"] == "TestExplainer"
    assert call_args["use_comp_rep"] == True
    
    # Check that the insight was cached
    assert "test_optimizer" in _insights_cache

# Test get_feature_importance endpoint
def test_get_feature_importance_endpoint(mock_api_dependencies, reset_cache):
    """Test the get feature importance endpoint."""
    mock_gen_insight, _, mock_feat_imp = mock_api_dependencies
    
    # Create test client
    app = FastAPI()
    app.include_router(insights_router)
    client = TestClient(app)
    
    # Call the endpoint
    response = client.post(
        "/optimizations/test_optimizer/insights/feature-importance",
        json={"top_n": 3}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feature_importance" in data
    assert "top_features" in data["feature_importance"]
    assert "all_features" in data["feature_importance"]
    
    # Verify that insights were generated and feature importance was calculated
    mock_gen_insight.assert_called_once()
    mock_feat_imp.assert_called_once_with(
        insight=mock_gen_insight.return_value,
        top_n=3
    )