#!/usr/bin/env python3
"""
Comprehensive Test Script for Bayesian Optimization API

This script tests all available endpoints in the API including:
- Main API endpoints
- Initialization endpoints
- Insights endpoints

Each test logs the result and any errors encountered.
"""

import requests
import json
import time
import pandas as pd
import numpy as np
import base64
import sys
from pprint import pprint
import argparse
import os
import io
from datetime import datetime

# API configuration
API_URL = "http://localhost:8000"
API_KEY = "123456789"  # Default API key from the code

# Configure headers for all requests
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total": 0
}

def print_header(text):
    """Print section header with formatting."""
    border = "=" * (len(text) + 4)
    print(f"\n{BLUE}{border}")
    print(f"= {text} =")
    print(f"{border}{RESET}\n")

def print_test_result(test_name, status, message="", response=None):
    """Print test result with formatting."""
    test_results["total"] += 1
    
    if status == "PASS":
        test_results["passed"] += 1
        status_color = GREEN
    elif status == "FAIL":
        test_results["failed"] += 1
        status_color = RED
    else:  # SKIP
        test_results["skipped"] += 1
        status_color = YELLOW
        
    print(f"{status_color}[{status}]{RESET} {test_name}")
    
    if message:
        print(f"       {message}")
        
    if response is not None and status == "FAIL":
        try:
            print(f"       Response: {response.status_code} - {response.text[:200]}...")
        except:
            print(f"       Response: {response}")

def make_request(method, endpoint, data=None, files=None, params=None, expected_status=200):
    """Make API request and handle errors."""
    url = f"{API_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=HEADERS, params=params)
        elif method.upper() == "POST":
            if files:
                # Don't include Content-Type header when uploading files
                file_headers = {k: v for k, v in HEADERS.items() if k != "Content-Type"}
                response = requests.post(url, headers=file_headers, data=data, files=files, params=params)
            else:
                response = requests.post(url, headers=HEADERS, json=data, params=params)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=HEADERS)
        else:
            return None, f"Unsupported method: {method}"
        
        if response.status_code != expected_status:
            return response, f"Expected status {expected_status}, got {response.status_code}"
        
        return response, None
    except Exception as e:
        return None, f"Request error: {str(e)}"

def handle_response(response, error, test_name):
    """Handle response and print test result."""
    if error:
        print_test_result(test_name, "FAIL", error, response)
        return False, None
    
    try:
        if response.text:
            data = response.json()
            if data.get("status") == "error":
                print_test_result(test_name, "FAIL", data.get("message", "Unknown error"), response)
                return False, data
            return True, data
        return True, {}
    except Exception as e:
        print_test_result(test_name, "FAIL", f"Failed to parse response: {str(e)}", response)
        return False, None

def test_health_check():
    """Test the health check endpoint."""
    response, error = make_request("GET", "/health")
    success, data = handle_response(response, error, "Health Check")
    
    if success:
        print_test_result("Health Check", "PASS", f"API is healthy, GPU: {data.get('using_gpu', False)}")
    
    return success

def test_create_optimization(optimizer_id):
    """Test creating an optimization process."""
    # Define test data for a chemical reaction optimization
    optimization_data = {
        "parameters": [
            {
                "type": "NumericalDiscrete",
                "name": "Temperature",
                "values": [50, 60, 70, 80, 90, 100],
                "tolerance": 0.5
            },
            {
                "type": "NumericalDiscrete",
                "name": "Pressure",
                "values": [1.0, 1.5, 2.0, 2.5, 3.0],
                "tolerance": 0.1
            },
            {
                "type": "CategoricalParameter",
                "name": "Catalyst",
                "values": ["A", "B", "C", "D"],
                "encoding": "OHE"
            }
        ],
        "target_config": {
            "name": "Yield",
            "mode": "MAX"
        },
        "recommender_config": {
            "type": "BotorchRecommender",
            "n_restarts": 10,
            "n_raw_samples": 64
        }
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/create", data=optimization_data)
    success, data = handle_response(response, error, "Create Optimization")
    
    if success:
        print_test_result("Create Optimization", "PASS", f"Created optimization with ID: {optimizer_id}")
    
    return success

def test_get_parameter_info(optimizer_id):
    """Test getting parameter information."""
    response, error = make_request("GET", f"/optimizations/{optimizer_id}/parameter-info")
    success, data = handle_response(response, error, "Get Parameter Info")
    
    if success:
        parameter_count = len(data.get("parameters", []))
        print_test_result("Get Parameter Info", "PASS", f"Got info for {parameter_count} parameters")
    
    return success

def test_initialize_with_predefined(optimizer_id):
    """Test initializing with predefined samples."""
    init_data = {
        "n_samples": 5,
        "seed": 42
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/initialize/predefined", data=init_data)
    success, data = handle_response(response, error, "Initialize with Predefined Samples")
    
    if success:
        sample_count = len(data.get("samples", []))
        print_test_result("Initialize with Predefined Samples", "PASS", f"Generated {sample_count} samples")
    
    return success

def test_suggest_next_point(optimizer_id):
    """Test getting next point suggestions."""
    response, error = make_request("GET", f"/optimizations/{optimizer_id}/suggest?batch_size=2")
    success, data = handle_response(response, error, "Suggest Next Points")
    
    if success:
        suggestion_count = len(data.get("suggestions", []))
        print_test_result("Suggest Next Points", "PASS", f"Got {suggestion_count} suggestions")
        return success, data.get("suggestions", [])
    
    return success, []

def test_add_measurement(optimizer_id, parameters=None):
    """Test adding a measurement."""
    # Default measurement if no parameters provided
    if parameters is None:
        measurement_data = {
            "parameters": {
                "Temperature": 70,
                "Pressure": 2.0,
                "Catalyst": "B"
            },
            "target_value": 82.5
        }
    else:
        measurement_data = {
            "parameters": parameters,
            "target_value": 85.0  # Simulated measurement
        }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/measurement", data=measurement_data)
    success, data = handle_response(response, error, "Add Measurement")
    
    if success:
        print_test_result("Add Measurement", "PASS", f"Added measurement with value: {measurement_data['target_value']}")
    
    return success

def test_add_multiple_measurements(optimizer_id):
    """Test adding multiple measurements at once."""
    measurements_data = {
        "measurements": [
            {"Temperature": 60, "Pressure": 1.5, "Catalyst": "A", "Yield": 75.2},
            {"Temperature": 80, "Pressure": 2.5, "Catalyst": "C", "Yield": 88.7},
            {"Temperature": 90, "Pressure": 3.0, "Catalyst": "D", "Yield": 79.3}
        ]
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/measurements", data=measurements_data)
    success, data = handle_response(response, error, "Add Multiple Measurements")
    
    if success:
        print_test_result("Add Multiple Measurements", "PASS", f"Added {len(measurements_data['measurements'])} measurements")
    
    return success

def test_get_best_point(optimizer_id):
    """Test getting the current best point."""
    response, error = make_request("GET", f"/optimizations/{optimizer_id}/best")
    success, data = handle_response(response, error, "Get Best Point")
    
    if success:
        if "best_value" in data:
            print_test_result("Get Best Point", "PASS", f"Best value: {data['best_value']}")
        else:
            print_test_result("Get Best Point", "PASS", "No measurements yet")
    
    return success

def test_get_measurement_history(optimizer_id):
    """Test getting measurement history."""
    response, error = make_request("GET", f"/optimizations/{optimizer_id}/history")
    success, data = handle_response(response, error, "Get Measurement History")
    
    if success:
        measurement_count = len(data.get("measurements", []))
        print_test_result("Get Measurement History", "PASS", f"Got history with {measurement_count} measurements")
    
    return success

def test_get_campaign_info(optimizer_id):
    """Test getting campaign information."""
    response, error = make_request("GET", f"/optimizations/{optimizer_id}/info")
    success, data = handle_response(response, error, "Get Campaign Info")
    
    if success:
        print_test_result("Get Campaign Info", "PASS", "Successfully retrieved campaign info")
    
    return success

def test_list_optimizations():
    """Test listing all optimizations."""
    response, error = make_request("GET", "/optimizations")
    success, data = handle_response(response, error, "List Optimizations")
    
    if success:
        optimizer_count = len(data.get("optimizers", []))
        print_test_result("List Optimizations", "PASS", f"Found {optimizer_count} optimizations")
    
    return success

def test_upload_csv(optimizer_id):
    """Test uploading CSV data."""
    # Create a CSV file in memory
    csv_content = "Temperature,Pressure,Catalyst,Yield\n70,2.0,A,78.5\n80,2.5,B,85.2\n90,3.0,C,81.7"
    
    # Prepare the file for upload
    files = {
        "file": ("data.csv", csv_content.encode('utf-8'), "text/csv")
    }
    
    # Add header=True as a query parameter
    params = {"header": "true"}
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/upload-csv", data=None, files=files, params=params)
    success, data = handle_response(response, error, "Upload CSV")
    
    if success:
        row_count = data.get("rows_parsed", 0)
        print_test_result("Upload CSV", "PASS", f"Uploaded CSV with {row_count} rows")
    
    return success

def test_generate_shap_insight(optimizer_id):
    """Test generating SHAP insights."""
    insight_data = {
        "explainer_type": "KernelExplainer",
        "use_comp_rep": False
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/insights/shap", data=insight_data)
    success, data = handle_response(response, error, "Generate SHAP Insight")
    
    if success:
        print_test_result("Generate SHAP Insight", "PASS", "SHAP insight generated successfully")
        return True
    else:
        # This might fail if there are too few measurements - that's expected sometimes
        print_test_result("Generate SHAP Insight", "SKIP", "Not enough measurements for SHAP analysis")
        return False

def test_generate_shap_plot(optimizer_id):
    """Test generating a SHAP plot."""
    # First try to generate SHAP insights if not already done
    success = test_generate_shap_insight(optimizer_id)
    if not success:
        print_test_result("Generate SHAP Plot", "SKIP", "SHAP insights not available")
        return False
    
    plot_data = {
        "plot_type": "bar",
        "plot_title": "Feature Importance"
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/insights/plot", data=plot_data)
    success, data = handle_response(response, error, "Generate SHAP Plot")
    
    if success:
        has_image = "image" in data
        print_test_result("Generate SHAP Plot", "PASS", "Plot generated successfully" if has_image else "Response received but no image")
    
    return success

def test_get_feature_importance(optimizer_id):
    """Test getting feature importance."""
    # First try to generate SHAP insights if not already done
    success = test_generate_shap_insight(optimizer_id)
    if not success:
        print_test_result("Get Feature Importance", "SKIP", "SHAP insights not available")
        return False
    
    importance_data = {
        "top_n": 3
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/insights/feature-importance", data=importance_data)
    success, data = handle_response(response, error, "Get Feature Importance")
    
    if success:
        has_importance = "feature_importance" in data
        print_test_result("Get Feature Importance", "PASS", "Feature importance retrieved successfully" if has_importance else "Response received but no feature importance")
    
    return success

def test_list_explainers():
    """Test listing available SHAP explainers."""
    response, error = make_request("GET", "/insights/explainers")
    success, data = handle_response(response, error, "List SHAP Explainers")
    
    if success:
        explainer_count = len(data) if isinstance(data, list) else 0
        print_test_result("List SHAP Explainers", "PASS", f"Found {explainer_count} explainers")
    
    return success

def test_list_plot_types():
    """Test listing available SHAP plot types."""
    response, error = make_request("GET", "/insights/plot-types")
    success, data = handle_response(response, error, "List SHAP Plot Types")
    
    if success:
        plot_type_count = len(data) if isinstance(data, list) else 0
        print_test_result("List SHAP Plot Types", "PASS", f"Found {plot_type_count} plot types")
    
    return success

def test_initialize_with_existing(optimizer_id):
    """Test initializing with existing data."""
    existing_data = {
        "data": {
            "Temperature": [60, 80, 90],
            "Pressure": [1.5, 2.5, 3.0],
            "Catalyst": ["A", "C", "D"],
            "Yield": [75.2, 88.7, 79.3]
        }
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/initialize/existing", data=existing_data)
    success, data = handle_response(response, error, "Initialize with Existing Data")
    
    if success:
        measurement_count = data.get("measurements_count", 0)
        print_test_result("Initialize with Existing Data", "PASS", f"Initialized with {measurement_count} measurements")
    
    return success

def test_initialize_from_zero(optimizer_id):
    """Test initializing from zero samples."""
    init_data = {
        "n_samples": 3,
        "seed": 123
    }
    
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/initialize/zero", data=init_data)
    success, data = handle_response(response, error, "Initialize from Zero")
    
    if success:
        sample_count = len(data.get("samples", []))
        print_test_result("Initialize from Zero", "PASS", f"Generated {sample_count} samples")
    
    return success

def test_delete_optimization(optimizer_id):
    """Test deleting an optimization."""
    response, error = make_request("DELETE", f"/optimizations/{optimizer_id}")
    success, data = handle_response(response, error, "Delete Optimization")
    
    if success:
        print_test_result("Delete Optimization", "PASS", f"Deleted optimization with ID: {optimizer_id}")
    
    return success

def test_load_optimization(optimizer_id):
    """Test loading an existing optimization."""
    response, error = make_request("POST", f"/optimizations/{optimizer_id}/load")
    success, data = handle_response(response, error, "Load Optimization")
    
    if success:
        print_test_result("Load Optimization", "PASS", f"Loaded optimization with ID: {optimizer_id}")
    
    return success

def run_full_test_suite():
    """Run all tests in sequence."""
    start_time = time.time()
    
    # Generate a unique ID for this test run to avoid conflicts
    timestamp = int(time.time())
    test_optimizer_id = f"api_test_{timestamp}"
    
    print_header("Starting Comprehensive API Test Suite")
    print(f"API URL: {API_URL}")
    print(f"Test Optimizer ID: {test_optimizer_id}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Core API Tests
    print_header("Core API Tests")
    if not test_health_check():
        print(f"{RED}API health check failed. Aborting tests.{RESET}")
        return False
    
    # Create optimization
    if not test_create_optimization(test_optimizer_id):
        print(f"{RED}Failed to create optimization. Aborting tests.{RESET}")
        return False
    
    # Test basic endpoints
    test_get_parameter_info(test_optimizer_id)
    
    # Test initialization endpoints
    print_header("Initialization Endpoints Tests")
    test_initialize_with_predefined(test_optimizer_id)
    test_initialize_from_zero(f"{test_optimizer_id}_zero")
    test_initialize_with_existing(f"{test_optimizer_id}_existing")
    test_upload_csv(test_optimizer_id)
    
    # Test recommendation and measurement endpoints
    print_header("Recommendation and Measurement Tests")
    success, suggestions = test_suggest_next_point(test_optimizer_id)
    
    if success and suggestions:
        # Add measurements using the suggestions
        for suggestion in suggestions:
            test_add_measurement(test_optimizer_id, suggestion)
    else:
        # Add a default measurement
        test_add_measurement(test_optimizer_id)
    
    # Add more measurements
    test_add_multiple_measurements(test_optimizer_id)
    
    # Test query endpoints
    test_get_best_point(test_optimizer_id)
    test_get_measurement_history(test_optimizer_id)
    test_get_campaign_info(test_optimizer_id)
    test_list_optimizations()
    
    # Test insights endpoints
    print_header("Insights Endpoints Tests")
    test_list_explainers()
    test_list_plot_types()
    test_generate_shap_insight(test_optimizer_id)
    test_generate_shap_plot(test_optimizer_id)
    test_get_feature_importance(test_optimizer_id)
    
    # Test management endpoints
    print_header("Management Endpoints Tests")
    test_delete_optimization(test_optimizer_id)
    
    # Create and then load
    test_create_optimization(f"{test_optimizer_id}_reload")
    test_load_optimization(f"{test_optimizer_id}_reload")
    
    # Print test summary
    end_time = time.time()
    elapsed = end_time - start_time
    
    print_header("Test Summary")
    print(f"Total Tests: {test_results['total']}")
    print(f"{GREEN}Passed: {test_results['passed']}{RESET}")
    print(f"{RED}Failed: {test_results['failed']}{RESET}")
    print(f"{YELLOW}Skipped: {test_results['skipped']}{RESET}")
    print(f"Elapsed Time: {elapsed:.2f} seconds")
    
    return test_results['failed'] == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Bayesian Optimization API endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL (default: http://localhost:8000)")
    parser.add_argument("--key", default="123456789", help="API Key (default: 123456789)")
    args = parser.parse_args()
    
    # Update global variables based on arguments
    API_URL = args.url
    API_KEY = args.key
    HEADERS["X-API-Key"] = API_KEY
    
    # Run the test suite
    success = run_full_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)