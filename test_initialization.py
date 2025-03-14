# test_initialization.py

import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, FastAPI

from initialization import (
    generate_samples,
    parse_existing_data,
    initialization_router
)

# Test fixtures
@pytest.fixture
def sample_parameters():
    """Sample parameters configuration for testing."""
    return [
        {
            "name": "x",
            "type": "NumericalContinuous",
            "bounds": [0, 10]
        },
        {
            "name": "y",
            "type": "NumericalDiscrete", 
            "values": [1, 2, 3, 4, 5]
        },
        {
            "name": "category",
            "type": "Categorical",
            "values": ["A", "B", "C"]
        }
    ]

@pytest.fixture
def sample_data():
    """Sample experimental data for testing."""
    return {
        "x": [1.0, 2.0, 5.0],
        "y": [1, 3, 5], 
        "category": ["A", "B", "C"],
        "target": [10.0, 20.0, 15.0]
    }

# Test generate_samples
def test_generate_samples_random(sample_parameters):
    """Test generating samples using random sampling."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test with random sampling
    samples = generate_samples(
        parameters=sample_parameters,
        n_samples=5,
        seed=42,
        use_lhs=False
    )
    
    # Check that the correct number of samples was generated
    assert len(samples) == 5
    
    # Check that all parameters are included
    assert set(samples.columns) == {"x", "y", "category"}
    
    # Check that numerical parameters are within bounds
    assert all(0 <= x <= 10 for x in samples["x"])
    assert all(y in [1, 2, 3, 4, 5] for y in samples["y"])
    assert all(c in ["A", "B", "C"] for c in samples["category"])

# Test with Latin Hypercube Sampling if available
def test_generate_samples_lhs(sample_parameters):
    """Test generating samples using LHS if available."""
    # Mock LHS availability
    with patch('initialization.LHS_AVAILABLE', True):
        with patch('initialization.lhs') as mock_lhs:
            # Configure mock to return a normalized LHS sample
            mock_lhs.return_value = np.array([
                [0.2, 0.3],
                [0.5, 0.7],
                [0.8, 0.1]
            ])
            
            # Generate samples
            samples = generate_samples(
                parameters=sample_parameters[:2],  # Just numerical parameters
                n_samples=3,
                seed=42,
                use_lhs=True
            )
            
            # Check results
            assert len(samples) == 3
            assert set(samples.columns) == {"x", "y"}
            mock_lhs.assert_called_once_with(2, samples=3)

# Test parse_existing_data
def test_parse_existing_data(sample_parameters, sample_data):
    """Test parsing existing experimental data."""
    # Parse the data
    df = parse_existing_data(
        data=sample_data,
        parameters=sample_parameters,
        target_name="target"
    )
    
    # Check that the dataframe has the correct structure
    assert len(df) == 3
    assert set(df.columns) == {"x", "y", "category", "target"}
    
    # Check that numerical columns are converted to float
    assert df["x"].dtype == np.float64
    assert df["y"].dtype == np.float64
    assert df["target"].dtype == np.float64
    
    # Check that categorical column is converted to string
    assert df["category"].dtype == object  # str in pandas

# Test validation in parse_existing_data
def test_parse_existing_data_validation(sample_parameters):
    """Test that validation works in parse_existing_data."""
    # Test missing parameter
    incomplete_data = {
        "x": [1.0, 2.0],
        # y is missing
        "category": ["A", "B"],
        "target": [10.0, 20.0]
    }
    
    with pytest.raises(ValueError, match="Missing parameters"):
        parse_existing_data(
            data=incomplete_data,
            parameters=sample_parameters,
            target_name="target"
        )
    
    # Test missing target
    no_target_data = {
        "x": [1.0, 2.0],
        "y": [1, 3],
        "category": ["A", "B"]
        # target is missing
    }
    
    with pytest.raises(ValueError, match="Target column"):
        parse_existing_data(
            data=no_target_data,
            parameters=sample_parameters,
            target_name="target"
        )

# Router endpoint tests - FIXED to properly mock the imports

@pytest.fixture
def mock_campaign():
    """Create a mock campaign for API testing."""
    mock = MagicMock()
    
    # Setup parameters with proper attributes
    param1 = MagicMock()
    param1.name = "x"
    param1.__class__.__name__ = "NumericalContinuousParameter"
    param1.bounds = MagicMock()
    param1.bounds.lower = 0.0
    param1.bounds.upper = 10.0
    param1.values = None
    
    param2 = MagicMock()
    param2.name = "y"
    param2.__class__.__name__ = "NumericalDiscreteParameter"
    param2.values = [1, 2, 3, 4, 5]
    param2.bounds = None
    
    param3 = MagicMock()
    param3.name = "category"
    param3.__class__.__name__ = "CategoricalParameter"
    param3.values = ["A", "B", "C"]
    param3.bounds = None
    
    mock.searchspace.parameters = [param1, param2, param3]
    
    # Setup target
    target = MagicMock()
    target.name = "target"
    target.mode = "MAX"
    mock.targets = [target]
    
    return mock

@pytest.fixture
def mock_api_dependencies(mock_campaign):
    """Setup mocks for API dependencies."""
    # Create mock for main.get_optimizer and main.bo_backend
    # We need to patch these at the module level where they're being used
    mock_get_optimizer = MagicMock(return_value=mock_campaign)
    mock_backend = MagicMock()
    mock_backend.add_multiple_measurements.return_value = {"status": "success"}
    
    # Create a module level patch
    sys.modules['main'] = MagicMock()
    sys.modules['main'].get_optimizer = mock_get_optimizer
    sys.modules['main'].bo_backend = mock_backend
    
    yield mock_backend
    
    # Clean up
    if 'main' in sys.modules:
        del sys.modules['main']

# Test initialize_with_predefined_samples endpoint
def test_initialize_with_predefined_samples_endpoint(mock_api_dependencies):
    """Test the predefined samples initialization endpoint."""
    # Create test client
    app = FastAPI()
    app.include_router(initialization_router)
    client = TestClient(app)
    
    # Call the endpoint
    with patch('initialization.generate_samples') as mock_generate:
        # Configure the mock to return a dataframe
        mock_generate.return_value = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1, 2, 3],
            "category": ["A", "B", "C"]
        })
        
        response = client.post(
            "/optimizations/test_optimizer/initialize/predefined",
            json={"n_samples": 3, "seed": 42}
        )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["samples"]) == 3
    assert data["samples"][0]["x"] == 1.0
    assert data["samples"][0]["y"] == 1
    assert data["samples"][0]["category"] == "A"

# Test initialize_with_existing_data endpoint
def test_initialize_with_existing_data_endpoint(mock_api_dependencies):
    """Test the existing data initialization endpoint."""
    # Create test client
    app = FastAPI()
    app.include_router(initialization_router)
    client = TestClient(app)
    
    # Prepare request data
    request_data = {
        "data": {
            "x": [1.0, 2.0, 3.0],
            "y": [1, 2, 3],
            "category": ["A", "B", "C"],
            "target": [10.0, 20.0, 15.0]
        }
    }
    
    # Call the endpoint
    with patch('initialization.parse_existing_data') as mock_parse:
        # Configure the mock to return a dataframe
        mock_parse.return_value = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1, 2, 3],
            "category": ["A", "B", "C"],
            "target": [10.0, 20.0, 15.0]
        })
        
        response = client.post(
            "/optimizations/test_optimizer/initialize/existing",
            json=request_data
        )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["measurements_count"] == 3
    
    # Verify that add_multiple_measurements was called
    mock_api_dependencies.add_multiple_measurements.assert_called_once()

# Test get_parameter_info endpoint
def test_get_parameter_info_endpoint(mock_api_dependencies):
    """Test the parameter info endpoint."""
    # Create test client
    app = FastAPI()
    app.include_router(initialization_router)
    client = TestClient(app)
    
    # Call the endpoint
    response = client.get("/optimizations/test_optimizer/parameter-info")
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["parameters"]) == 3
    assert data["parameters"][0]["name"] == "x"
    assert data["parameters"][1]["name"] == "y"
    assert data["parameters"][2]["name"] == "category"
    assert data["target"]["name"] == "target"
    assert data["target"]["mode"] == "MAX"

# Test upload_csv_data endpoint
def test_upload_csv_data_endpoint(mock_api_dependencies):
    """Test the CSV upload endpoint."""
    # Create test client
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(initialization_router)
    client = TestClient(app)
    
    # Prepare CSV content
    csv_content = "x,y,category,target\n1.0,1,A,10.0\n2.0,2,B,20.0\n3.0,3,C,15.0"
    
    # Call the endpoint with mocked read_csv
    with patch('pandas.read_csv') as mock_read_csv:
        # Configure the mock to return a dataframe
        mock_read_csv.return_value = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1, 2, 3],
            "category": ["A", "B", "C"],
            "target": [10.0, 20.0, 15.0]
        })
        
        # Use files parameter for multipart/form-data file upload
        # This is the standard way to upload files with FastAPI
        response = client.post(
            "/optimizations/test_optimizer/upload-csv",
            files={"file": ("data.csv", csv_content.encode('utf-8'), "text/csv")},
            params={"header": "true"}
        )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["rows_parsed"] == 3
    assert set(data["parameters"]) == {"x", "y", "category"}
    
    # Verify that add_multiple_measurements was called
    mock_api_dependencies.add_multiple_measurements.assert_called_once()