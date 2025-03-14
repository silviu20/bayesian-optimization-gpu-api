# test_main.py

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app, get_optimizer, bo_backend

# Test client
client = TestClient(app)

# Setup mocks for the backend
@pytest.fixture
def mock_backend():
    """Set up mocked backend for testing."""
    with patch('main.bo_backend') as mock_bo:
        # Setup default responses for common methods
        mock_bo.create_optimization.return_value = {"status": "success", "optimizer_id": "test_optimizer"}
        mock_bo.suggest_next_point.return_value = {"status": "success", "suggestions": [{"x": 5.0, "y": 2}]}
        mock_bo.add_measurement.return_value = {"status": "success", "message": "Measurement added"}
        mock_bo.add_multiple_measurements.return_value = {"status": "success", "message": "Measurements added"}
        mock_bo.get_best_point.return_value = {"status": "success", "best_parameters": {"x": 5.0, "y": 2}, "best_value": 25.0}
        mock_bo.load_campaign.return_value = {"status": "success", "message": "Campaign loaded"}
        mock_bo.get_measurement_history.return_value = {"status": "success", "measurements": []}
        mock_bo.get_campaign_info.return_value = {"status": "success", "info": {}}
        mock_bo.list_optimizations.return_value = {"status": "success", "optimizers": []}
        
        # Setup active_campaigns attribute
        mock_bo.active_campaigns = {"test_optimizer": MagicMock()}
        
        # Setup device attribute for health check
        mock_bo.device = MagicMock()
        mock_bo.device.type = "cpu"
        
        yield mock_bo

# Test health check endpoint
def test_health_check(mock_backend):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "using_gpu" in data
    
    # Test with GPU
    mock_backend.device.type = "cuda"
    with patch('torch.cuda.get_device_name', return_value="Test GPU"):
        with patch('torch.cuda.memory_allocated', return_value=0):
            with patch('torch.cuda.memory_cached', return_value=0):
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["using_gpu"] == True
                assert data["gpu_info"]["name"] == "Test GPU"

# Test create_optimization endpoint
def test_create_optimization(mock_backend):
    """Test creating an optimization process."""
    # Prepare request data
    optimizer_id = "test_optimizer"
    request_data = {
        "parameters": [
            {"name": "x", "type": "NumericalContinuous", "bounds": [0, 10]},
            {"name": "y", "type": "NumericalDiscrete", "values": [1, 2, 3]}
        ],
        "target_config": {"name": "target", "mode": "MAX"}
    }
    
    # Make request
    response = client.post(f"/optimizations/{optimizer_id}/create", json=request_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["optimizer_id"] == optimizer_id
    
    # Verify backend call
    mock_backend.create_optimization.assert_called_once_with(
        optimizer_id=optimizer_id,
        parameters=request_data["parameters"],
        target_config=request_data["target_config"],
        constraints=None,
        recommender_config=None,
        objective_type="SingleTarget",
        surrogate_config=None,
        acquisition_config=None
    )
    
    # Test error case
    mock_backend.create_optimization.return_value = {"status": "error", "message": "Test error"}
    response = client.post(f"/optimizations/{optimizer_id}/create", json=request_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

# Test suggest_next_point endpoint
def test_suggest_next_point(mock_backend):
    """Test suggesting next points to evaluate."""
    # Make request
    response = client.get("/optimizations/test_optimizer/suggest?batch_size=2")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "suggestions" in data
    
    # Verify backend call
    mock_backend.suggest_next_point.assert_called_once_with(
        optimizer_id="test_optimizer",
        batch_size=2
    )
    
    # Test error case when optimizer doesn't exist
    mock_backend.active_campaigns = {}
    mock_backend.load_campaign.return_value = {"status": "error", "message": "Optimizer not found"}
    response = client.get("/optimizations/nonexistent/suggest")
    assert response.status_code == 404

# Test add_measurement endpoint
def test_add_measurement(mock_backend):
    """Test adding a new measurement."""
    # Prepare request data
    request_data = {
        "parameters": {"x": 5.0, "y": 2},
        "target_value": 25.0
    }
    
    # Make request
    response = client.post("/optimizations/test_optimizer/measurement", json=request_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Verify backend call
    mock_backend.add_measurement.assert_called_once_with(
        optimizer_id="test_optimizer",
        parameters=request_data["parameters"],
        target_value=request_data["target_value"]
    )
    
    # Test error case
    mock_backend.add_measurement.return_value = {"status": "error", "message": "Test error"}
    response = client.post("/optimizations/test_optimizer/measurement", json=request_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

# Test add_multiple_measurements endpoint
def test_add_multiple_measurements(mock_backend):
    """Test adding multiple measurements at once."""
    # Prepare request data
    request_data = {
        "measurements": [
            {"x": 1.0, "y": 1, "target": 10.0},
            {"x": 2.0, "y": 2, "target": 20.0},
            {"x": 3.0, "y": 3, "target": 15.0}
        ]
    }
    
    # Make request
    response = client.post("/optimizations/test_optimizer/measurements", json=request_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Verify backend call - note we can't easily check the DataFrame contents
    mock_backend.add_multiple_measurements.assert_called_once()
    
    # Test error case
    mock_backend.add_multiple_measurements.return_value = {"status": "error", "message": "Test error"}
    response = client.post("/optimizations/test_optimizer/measurements", json=request_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

# Test get_best_point endpoint
def test_get_best_point(mock_backend):
    """Test getting the current best point."""
    # Make request
    response = client.get("/optimizations/test_optimizer/best")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["best_parameters"] == {"x": 5.0, "y": 2}
    assert data["best_value"] == 25.0
    
    # Verify backend call
    mock_backend.get_best_point.assert_called_once_with(optimizer_id="test_optimizer")
    
    # Test error case
    mock_backend.get_best_point.return_value = {"status": "error", "message": "Test error"}
    response = client.get("/optimizations/test_optimizer/best")
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

# Test load_optimization endpoint
def test_load_optimization(mock_backend):
    """Test loading an existing optimization."""
    # Make request
    response = client.post("/optimizations/test_optimizer/load")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Verify backend call
    mock_backend.load_campaign.assert_called_once_with("test_optimizer")
    
    # Test error case
    mock_backend.load_campaign.return_value = {"status": "error", "message": "Optimizer not found"}
    response = client.post("/optimizations/nonexistent/load")
    assert response.status_code == 404
    assert response.json()["detail"] == "Optimizer not found"

# Test get_measurement_history endpoint
def test_get_measurement_history(mock_backend):
    """Test getting the measurement history."""
    # Make request
    response = client.get("/optimizations/test_optimizer/history")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "measurements" in data
    
    # Verify backend call
    mock_backend.get_measurement_history.assert_called_once_with(optimizer_id="test_optimizer")
    
    # Test error case
    mock_backend.get_measurement_history.return_value = {"status": "error", "message": "Test error"}
    response = client.get("/optimizations/test_optimizer/history")
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

# Test get_campaign_info endpoint
def test_get_campaign_info(mock_backend):
    """Test getting campaign information."""
    # Make request
    response = client.get("/optimizations/test_optimizer/info")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "info" in data
    
    # Verify backend call
    mock_backend.get_campaign_info.assert_called_once_with(optimizer_id="test_optimizer")
    
    # Test error case
    mock_backend.get_campaign_info.return_value = {"status": "error", "message": "Test error"}
    response = client.get("/optimizations/test_optimizer/info")
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

# Test list_optimizations endpoint
def test_list_optimizations(mock_backend):
    """Test listing all available optimizations."""
    # Configure mock to return some optimizers
    mock_backend.list_optimizations.return_value = {
        "status": "success",
        "optimizers": [
            {"id": "optimizer1", "file_path": "/path/to/optimizer1.json"},
            {"id": "optimizer2", "file_path": "/path/to/optimizer2.json"}
        ]
    }
    
    # Make request
    response = client.get("/optimizations")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["optimizers"]) == 2
    assert data["optimizers"][0]["id"] == "optimizer1"
    assert data["optimizers"][1]["id"] == "optimizer2"
    
    # Verify backend call
    mock_backend.list_optimizations.assert_called_once()
    
    # Test error case
    mock_backend.list_optimizations.return_value = {"status": "error", "message": "Test error"}
    response = client.get("/optimizations")
    assert response.status_code == 500
    assert response.json()["detail"] == "Test error"

# Test delete_optimization endpoint
def test_delete_optimization(mock_backend):
    """Test deleting an optimization from memory."""
    # Make request to delete existing optimizer
    mock_backend.active_campaigns = {"test_optimizer": MagicMock()}
    response = client.delete("/optimizations/test_optimizer")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "test_optimizer" not in mock_backend.active_campaigns
    
    # Test deleting non-existent optimizer
    response = client.delete("/optimizations/nonexistent")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warning"

# Test get_optimizer helper function
def test_get_optimizer(mock_backend):
    """Test the get_optimizer helper function."""
    # Test retrieving existing optimizer
    mock_backend.active_campaigns = {"test_optimizer": "test_value"}
    optimizer = get_optimizer("test_optimizer")
    assert optimizer == "test_value"
    
    # Test trying to load missing optimizer
    mock_backend.active_campaigns = {}
    mock_backend.load_campaign.return_value = {"status": "error", "message": "Optimizer not found"}
    with pytest.raises(Exception):
        get_optimizer("nonexistent")
        
    # Test successfully loading optimizer
    mock_backend.load_campaign.return_value = {"status": "success", "message": "Optimizer loaded"}
    mock_backend.active_campaigns = {"loaded_optimizer": "loaded_value"}
    optimizer = get_optimizer("loaded_optimizer")
    assert optimizer == "loaded_value"
