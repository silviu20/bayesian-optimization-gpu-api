#!/usr/bin/env python3
"""
Test client for the Bayesian Optimization API, demonstrating real-world usage.
This example optimizes a simulated chemistry reaction with multiple parameters.
"""

import requests
import json
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from typing import Dict, List, Any, Tuple

API_URL = "http://localhost:8000"
API_KEY = "123456789"  # Same as in .env file

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Function to test GPU health endpoint
def test_health():
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health check status code: {response.status_code}")
        health_data = response.json()
        print("API Health Status:")
        print(f"  Status: {health_data['status']}")
        print(f"  Using GPU: {health_data['using_gpu']}")
        
        if health_data['gpu_info']:
            print("GPU Information:")
            print(f"  GPU: {health_data['gpu_info']['name']}")
            print(f"  Memory Allocated: {health_data['gpu_info']['memory_allocated_mb']:.2f} MB")
            print(f"  Memory Reserved: {health_data['gpu_info']['memory_reserved_mb']:.2f} MB")
        else:
            print("No GPU information available (running in CPU mode)")
        
        return health_data['using_gpu']
    except Exception as e:
        print(f"Health check error: {e}")
        return False

# Function to create an optimization campaign
def create_optimization(optimizer_id: str) -> bool:
    """Create a chemical reaction optimization with multiple parameters."""
    payload = {
        "parameters": [
            {
                "type": "NumericalDiscrete",
                "name": "Temperature",
                "values": [50, 60, 70, 80, 90, 100, 110, 120],
                "tolerance": 0.5
            },
            {
                "type": "NumericalDiscrete",
                "name": "Pressure",
                "values": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                "tolerance": 0.1
            },
            {
                "type": "NumericalDiscrete",
                "name": "Time",
                "values": [30, 45, 60, 75, 90, 105, 120],
                "tolerance": 1.0
            },
            {
                "type": "CategoricalParameter",
                "name": "Catalyst",
                "values": ["A", "B", "C", "D", "E"],
                "encoding": "OHE"
            },
            {
                "type": "CategoricalParameter",
                "name": "Solvent",
                "values": ["Water", "Methanol", "Ethanol", "Acetone", "THF"],
                "encoding": "OHE"
            }
        ],
        "target_config": {
            "name": "Yield",
            "mode": "MAX"
        },
        "recommender_config": {
            "type": "BotorchRecommender",
            "n_restarts": 20,
            "n_raw_samples": 128
        }
    }

    try:
        response = requests.post(
            f"{API_URL}/optimization/{optimizer_id}",
            headers=headers,
            json=payload
        )
        print(f"Optimization creation status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Optimization created successfully!")
            result = response.json()
            pprint(result)
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Optimization creation error: {e}")
        return False

# Function to simulate a chemical reaction yield
def simulate_yield(temperature: float, pressure: float, time: float, catalyst: str, solvent: str) -> float:
    """
    Simulate a chemical reaction yield based on parameters.
    This is a complex model with interactions between parameters.
    """
    # Base effects for catalysts (with catalyst B being the best)
    catalyst_effect = {
        "A": 0.75,
        "B": 1.0,
        "C": 0.85,
        "D": 0.60,
        "E": 0.90
    }
    
    # Base effects for solvents (with THF being the best)
    solvent_effect = {
        "Water": 0.65,
        "Methanol": 0.80,
        "Ethanol": 0.85,
        "Acetone": 0.90,
        "THF": 0.95
    }
    
    # Temperature effect (parabolic with optimum around 90°C)
    temp_effect = -0.01 * (temperature - 90)**2 + 1.0
    
    # Pressure effect (generally higher is better but diminishing returns)
    pressure_effect = 0.1 * np.log(pressure) + 0.8
    
    # Reaction time effect (longer is generally better but with diminishing returns)
    time_effect = 0.2 * np.log(time/30) + 0.7
    
    # Special interaction effects
    interaction_effects = 0.0
    
    # Catalyst B works especially well with Ethanol
    if catalyst == "B" and solvent == "Ethanol":
        interaction_effects += 0.1
    
    # Catalyst E works especially well with Acetone
    if catalyst == "E" and solvent == "Acetone":
        interaction_effects += 0.15
    
    # High temperatures work poorly with Water as solvent
    if solvent == "Water" and temperature > 90:
        interaction_effects -= 0.2
    
    # Calculate base yield
    base_yield = (
        catalyst_effect[catalyst] * 
        solvent_effect[solvent] * 
        temp_effect * 
        pressure_effect * 
        time_effect
    )
    
    # Add interaction effects and scale to percentage
    raw_yield = (base_yield + interaction_effects) * 100
    
    # Add some randomness to simulate experimental variability
    yield_value = raw_yield + random.uniform(-3, 3)
    
    # Constrain to reasonable values
    return max(5, min(98, yield_value))

# Function to get a recommendation from the API
def get_recommendation(optimizer_id: str, batch_size: int = 1) -> List[Dict]:
    try:
        response = requests.get(
            f"{API_URL}/optimization/{optimizer_id}/suggest?batch_size={batch_size}",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Failed to get recommendation: {response.text}")
            return []
            
        recommendation = response.json()
        suggestions = recommendation["suggestions"]
        return suggestions
    except Exception as e:
        print(f"Error getting recommendation: {e}")
        return []

# Function to add a measurement to the API
def add_measurement(optimizer_id: str, parameters: Dict[str, Any], result_value: float) -> bool:
    measurement = {
        "parameters": parameters,
        "target_value": result_value
    }
    
    try:
        response = requests.post(
            f"{API_URL}/optimization/{optimizer_id}/measurement",
            headers=headers,
            json=measurement
        )
        
        if response.status_code != 200:
            print(f"Failed to add measurement: {response.text}")
            return False
        
        return True
    except Exception as e:
        print(f"Error adding measurement: {e}")
        return False

# Function to get the current best point
def get_best_point(optimizer_id: str) -> Dict:
    try:
        response = requests.get(
            f"{API_URL}/optimization/{optimizer_id}/best",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Failed to get best point: {response.text}")
            return {}
            
        result = response.json()
        return result
    except Exception as e:
        print(f"Error getting best point: {e}")
        return {}

# Function to add initial measurements
def add_initial_measurements(optimizer_id: str, num_initial: int = 5) -> List[Dict]:
    """Add initial measurements to bootstrap the Bayesian optimization."""
    print(f"\n=== Adding {num_initial} Initial Measurements ===\n")
    initial_experiments = []
    
    # Define some diverse initial points to explore the parameter space
    initial_params = [
        {"Temperature": 70, "Pressure": 2.0, "Time": 60, "Catalyst": "B", "Solvent": "Ethanol"},
        {"Temperature": 90, "Pressure": 3.0, "Time": 90, "Catalyst": "C", "Solvent": "Acetone"},
        {"Temperature": 50, "Pressure": 1.5, "Time": 45, "Catalyst": "A", "Solvent": "Water"},
        {"Temperature": 110, "Pressure": 4.0, "Time": 120, "Catalyst": "E", "Solvent": "THF"},
        {"Temperature": 80, "Pressure": 2.5, "Time": 75, "Catalyst": "D", "Solvent": "Methanol"}
    ]
    
    # Use as many points as requested (but not more than available)
    for i in range(min(num_initial, len(initial_params))):
        params = initial_params[i]
        print(f"\nInitial point {i+1}/{num_initial}:")
        print(f"Parameters: {params}")
        
        # Simulate running the experiment
        temperature = params["Temperature"]
        pressure = params["Pressure"]
        reaction_time = params["Time"]
        catalyst = params["Catalyst"]
        solvent = params["Solvent"]
        
        yield_value = simulate_yield(temperature, pressure, reaction_time, catalyst, solvent)
        print(f"Measured yield: {yield_value:.2f}%")
        
        # Add measurement back to the campaign
        success = add_measurement(optimizer_id, params, yield_value)
        if not success:
            print("Failed to add measurement, stopping.")
            break
        
        # Store experiment results
        initial_experiments.append({
            "iteration": i+1,
            "temperature": temperature,
            "pressure": pressure,
            "time": reaction_time,
            "catalyst": catalyst,
            "solvent": solvent,
            "yield": yield_value
        })
        
        # Add a small delay to avoid hammering the API
        time.sleep(0.2)
    
    return initial_experiments

# Function to run the optimization loop
def run_optimization(optimizer_id: str, num_iterations: int = 20, initial_experiments: List[Dict] = None) -> List[Dict]:
    print(f"\n=== Running {num_iterations} Bayesian Optimization Iterations ===\n")
    all_experiments = initial_experiments or []
    
    # Start iteration count after initial experiments
    start_iteration = len(all_experiments) + 1
    
    for i in range(num_iterations):
        iteration_num = start_iteration + i
        print(f"\nIteration {iteration_num}/{start_iteration + num_iterations - 1}:")
        
        # 1. Get next recommendation
        suggestions = get_recommendation(optimizer_id)
        if not suggestions:
            print("No recommendations received, stopping.")
            break
        
        params = suggestions[0]
        print(f"Recommended parameters: {params}")
        
        # 2. Simulate running the experiment
        temperature = params["Temperature"]
        pressure = params["Pressure"]
        reaction_time = params["Time"]
        catalyst = params["Catalyst"]
        solvent = params["Solvent"]
        
        yield_value = simulate_yield(temperature, pressure, reaction_time, catalyst, solvent)
        print(f"Measured yield: {yield_value:.2f}%")
        
        # 3. Add measurement back to the campaign
        success = add_measurement(optimizer_id, params, yield_value)
        if not success:
            print("Failed to add measurement, stopping.")
            break
        
        # 4. Store experiment results
        all_experiments.append({
            "iteration": iteration_num,
            "temperature": temperature,
            "pressure": pressure,
            "time": reaction_time,
            "catalyst": catalyst,
            "solvent": solvent,
            "yield": yield_value
        })
        
        # Add a small delay to avoid hammering the API
        time.sleep(0.2)
    
    return all_experiments

# Function to plot the optimization results
def plot_results(experiments: List[Dict]):
    # Extract data for plotting
    iterations = [exp["iteration"] for exp in experiments]
    yields = [exp["yield"] for exp in experiments]
    
    # Calculate best yield at each iteration (cumulative max)
    best_yields = [max(yields[:i+1]) for i in range(len(yields))]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Yield vs. Iteration
    ax1.plot(iterations, yields, 'o-', color='blue', label='Measured Yield')
    ax1.plot(iterations, best_yields, 's-', color='red', label='Best Yield')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Yield (%)')
    ax1.set_title('Yield vs. Iteration')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Parameter exploration over iterations
    # For numerical parameters
    ax2.scatter(iterations, [exp["temperature"] for exp in experiments], 
               marker='o', label='Temperature (°C)')
    ax2.scatter(iterations, [exp["pressure"]*20 for exp in experiments], 
               marker='s', label='Pressure*20 (bar)')
    ax2.scatter(iterations, [exp["time"]/2 for exp in experiments], 
               marker='^', label='Time/2 (min)')
    
    # Add text annotations for categorical parameters
    for i, exp in enumerate(experiments):
        ax2.annotate(f"{exp['catalyst']},{exp['solvent'][0]}", 
                    (iterations[i], 10), 
                    fontsize=8)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Exploration')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('optimization_results.png')
    print("Results plot saved as 'optimization_results.png'")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("=== Bayesian Optimization API Test Client ===")
    
    # Check if the API is healthy and using GPU
    using_gpu = test_health()
    print(f"\nAPI using GPU acceleration: {using_gpu}")
    
    # Generate a unique optimizer ID
    optimizer_id = f"chemical-reaction-{int(time.time())}"
    print(f"\nCreating optimization with ID: {optimizer_id}")
    
    # Create the optimization
    if create_optimization(optimizer_id):
        # Add initial measurements first (before requesting recommendations)
        initial_experiments = add_initial_measurements(optimizer_id, num_initial=5)
        
        # Then run the optimization loop
        all_experiments = run_optimization(
            optimizer_id, 
            num_iterations=15,  # Reduced from 20 to 15 since we're adding 5 initial points
            initial_experiments=initial_experiments
        )
        
        # Print summary of all experiments
        print("\n=== Experiment Summary ===")
        print(f"{'Iter #':<6} {'Temp (°C)':<10} {'Pressure (bar)':<14} {'Time (min)':<10} {'Catalyst':<10} {'Solvent':<10} {'Yield (%)':<10}")
        print("-" * 75)
        for exp in all_experiments:
            print(f"{exp['iteration']:<6} {exp['temperature']:<10} {exp['pressure']:<14} {exp['time']:<10} {exp['catalyst']:<10} {exp['solvent']:<10} {exp['yield']:.2f}")
        
        # Get the best result
        best = get_best_point(optimizer_id)
        if best and "best_parameters" in best:
            print("\n=== Best Result ===")
            print(f"Best parameters: {best['best_parameters']}")
            print(f"Best yield: {best['best_value']:.2f}%")
            
            # Plot the results
            plot_results(all_experiments)
    
    print("\nTest client execution completed.")