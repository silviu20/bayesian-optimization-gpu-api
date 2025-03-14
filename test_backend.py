# test_backend.py

import pytest
import json
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from backend import BayesianOptimizationBackend

# Test fixtures
@pytest.fixture
def mock_campaign():
    """Create a mock campaign object."""
    mock = MagicMock()
    mock.targets = [MagicMock()]
    mock.targets[0].name = "test_target"
    mock.targets[0].mode = "MAX"
    mock.searchspace.parameters = []
    mock.measurements = pd.DataFrame()
    mock.to_json.return_value = json.dumps({"mock": "campaign"})
    return mock

@pytest.fixture
def backend():
    """Create a test backend instance with mocked GPU."""
    with patch('torch.cuda.is_available', return_value=False):
        backend = BayesianOptimizationBackend(storage_path="./test_data", use_gpu=False)
    return backend

@pytest.fixture
def backend_with_campaign(backend, mock_campaign):
    """Create a backend with a mock campaign."""
    backend.active_campaigns = {"test_optimizer": mock_campaign}
    return backend

# Test initialization
def test_initialization():
    """Test backend initialization with both CPU and GPU options."""
    # Test CPU initialization
    with patch('torch.cuda.is_available', return_value=False):
        backend = BayesianOptimizationBackend(storage_path="./test_data", use_gpu=False)
        assert backend.device.type == "cpu"
        assert backend.storage_path == Path("./test_data")
        assert backend.active_campaigns == {}
    
    # Test GPU initialization when available
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.get_device_name', return_value="Test GPU"):
            with patch('torch.cuda.memory_allocated', return_value=0):
                with patch('torch.cuda.memory_cached', return_value=0):
                    with patch('torch.set_default_tensor_type'):
                        backend = BayesianOptimizationBackend(storage_path="./test_data", use_gpu=True)
                        assert backend.device.type == "cuda"

# Test create_optimization
def test_create_optimization(backend):
    """Test creating an optimization campaign."""
    # Mock the Campaign class
    with patch('backend.Campaign') as MockCampaign:
        # Configure mocks
        mock_campaign = MagicMock()
        MockCampaign.return_value = mock_campaign
        
        # Set up test data
        optimizer_id = "test_optimizer"
        parameters = [
            {"name": "x", "type": "NumericalContinuous", "bounds": [0, 10]},
            {"name": "y", "type": "NumericalDiscrete", "values": [1, 2, 3]}
        ]
        target_config = {"name": "target", "mode": "MAX"}
        
        # Call the method
        with patch('backend.Path.mkdir'):
            with patch('builtins.open', mock_open()):
                result = backend.create_optimization(
                    optimizer_id=optimizer_id,
                    parameters=parameters,
                    target_config=target_config
                )
        
        # Verify results
        assert result["status"] == "success"
        assert result["optimizer_id"] == optimizer_id
        assert optimizer_id in backend.active_campaigns
        
# Test suggest_next_point
def test_suggest_next_point(backend_with_campaign):
    """Test suggesting next points for evaluation."""
    # Configure the mock campaign
    mock_campaign = backend_with_campaign.active_campaigns["test_optimizer"]
    mock_campaign.recommend.return_value = pd.DataFrame({
        "x": [5.0],
        "y": [2]
    })
    
    # Call the method
    result = backend_with_campaign.suggest_next_point(
        optimizer_id="test_optimizer",
        batch_size=1
    )
    
    # Verify results
    assert result["status"] == "success"
    assert "suggestions" in result
    assert isinstance(result["suggestions"], list)
    assert len(result["suggestions"]) == 1
    assert "x" in result["suggestions"][0]
    assert "y" in result["suggestions"][0]

# Test add_measurement
def test_add_measurement(backend_with_campaign):
    """Test adding a measurement."""
    # Configure the mock campaign
    mock_campaign = backend_with_campaign.active_campaigns["test_optimizer"]
    
    # Set up test data
    parameters = {"x": 5.0, "y": 2}
    target_value = 10.0
    
    # Call the method with mocked save_campaign
    with patch.object(backend_with_campaign, 'save_campaign', return_value={"status": "success"}):
        result = backend_with_campaign.add_measurement(
            optimizer_id="test_optimizer",
            parameters=parameters,
            target_value=target_value
        )
    
    # Verify results
    assert result["status"] == "success"
    mock_campaign.add_measurements.assert_called_once()
    # Verify the DataFrame passed to add_measurements has the right structure
    df_arg = mock_campaign.add_measurements.call_args[0][0]
    assert df_arg.shape[0] == 1
    assert "x" in df_arg.columns
    assert "y" in df_arg.columns
    assert "test_target" in df_arg.columns  # Target name from the mock

# Test add_multiple_measurements
def test_add_multiple_measurements(backend_with_campaign):
    """Test adding multiple measurements at once."""
    # Configure the mock campaign
    mock_campaign = backend_with_campaign.active_campaigns["test_optimizer"]
    
    # Set up test data
    measurements = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1, 2, 3],
        "test_target": [10.0, 20.0, 30.0]  # Target name from the mock
    })
    
    # Call the method with mocked save_campaign
    with patch.object(backend_with_campaign, 'save_campaign', return_value={"status": "success"}):
        result = backend_with_campaign.add_multiple_measurements(
            optimizer_id="test_optimizer",
            measurements=measurements
        )
    
    # Verify results
    assert result["status"] == "success"
    mock_campaign.add_measurements.assert_called_once_with(measurements)

# Test get_best_point
def test_get_best_point(backend_with_campaign):
    """Test getting the current best point."""
    # Configure the mock campaign
    mock_campaign = backend_with_campaign.active_campaigns["test_optimizer"]
    mock_campaign.measurements = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1, 2, 3],
        "test_target": [10.0, 30.0, 20.0]  # Target name from the mock
    })
    
    # Call the method
    result = backend_with_campaign.get_best_point(optimizer_id="test_optimizer")
    
    # Verify results
    assert result["status"] == "success"
    assert result["best_value"] == 30.0  # Max value
    assert result["best_parameters"] == {"x": 2.0, "y": 2}
    assert result["total_measurements"] == 3

# Test save_campaign
def test_save_campaign(backend_with_campaign):
    """Test saving campaign to disk."""
    # Call the method
    with patch('builtins.open', mock_open()) as mock_file:
        result = backend_with_campaign.save_campaign(optimizer_id="test_optimizer")
    
    # Verify results
    assert result["status"] == "success"
    mock_file.assert_called_once()
    mock_file().write.assert_called_once()

# Test load_campaign
def test_load_campaign(backend):
    """Test loading a campaign from disk."""
    # Set up mocks
    mock_json = json.dumps({"mock": "campaign"})
    
    # Mock Campaign.from_json and Path.exists
    with patch('backend.Campaign.from_json') as mock_from_json:
        with patch('backend.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_json)):
                # Call the method
                result = backend.load_campaign(optimizer_id="test_optimizer")
    
    # Verify results
    assert result["status"] == "success"
    assert "test_optimizer" in backend.active_campaigns
    mock_from_json.assert_called_once_with(mock_json)

# Test get_measurement_history
def test_get_measurement_history(backend_with_campaign):
    """Test getting measurement history."""
    # Configure the mock campaign
    mock_campaign = backend_with_campaign.active_campaigns["test_optimizer"]
    mock_campaign.measurements = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [1, 2, 3],
        "test_target": [10.0, 20.0, 30.0]
    })
    
    # Call the method
    result = backend_with_campaign.get_measurement_history(optimizer_id="test_optimizer")
    
    # Verify results
    assert result["status"] == "success"
    assert "measurements" in result
    assert isinstance(result["measurements"], list)
    assert len(result["measurements"]) == 3

# Test get_campaign_info
def test_get_campaign_info(backend_with_campaign):
    """Test getting campaign information."""
    # Configure the mock campaign with more detailed structure
    mock_campaign = backend_with_campaign.active_campaigns["test_optimizer"]
    
    # Setup parameters with proper attributes
    param1 = MagicMock()
    param1.name = "x"
    param1.__class__.__name__ = "NumericalContinuousParameter"
    param1.bounds.lower = 0.0
    param1.bounds.upper = 10.0
    param1.values = None
    
    param2 = MagicMock()
    param2.name = "y"
    param2.__class__.__name__ = "NumericalDiscreteParameter"
    param2.values = [1, 2, 3]
    param2.bounds = None
    
    mock_campaign.searchspace.parameters = [param1, param2]
    mock_campaign.measurements = pd.DataFrame({"x": [1.0], "y": [1], "test_target": [10.0]})
    
    # Call the method
    result = backend_with_campaign.get_campaign_info(optimizer_id="test_optimizer")
    
    # Verify results
    assert result["status"] == "success"
    assert "info" in result
    assert "parameters" in result["info"]
    assert "target" in result["info"]
    assert result["info"]["parameters"][0]["name"] == "x"
    assert result["info"]["parameters"][1]["name"] == "y"
    assert result["info"]["target"]["name"] == "test_target"
    assert result["info"]["target"]["mode"] == "MAX"

# Test list_optimizations
def test_list_optimizations(backend):
    """Test listing all available optimizations."""
    # Mock Path.glob to return some JSON files
    with patch('backend.Path.glob') as mock_glob:
        mock_glob.return_value = [
            Path("./test_data/optimizer1.json"),
            Path("./test_data/optimizer2.json")
        ]
        
        # Mock open to return valid campaign content
        mock_content = '{"searchspace": {}, "objective": {}}'
        with patch('builtins.open', mock_open(read_data=mock_content)):
            # Call the method
            result = backend.list_optimizations()
    
    # Verify results
    assert result["status"] == "success"
    assert "optimizers" in result
    assert len(result["optimizers"]) == 2
    assert result["optimizers"][0]["id"] == "optimizer1"
    assert result["optimizers"][1]["id"] == "optimizer2"