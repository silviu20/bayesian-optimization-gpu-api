#!/usr/bin/env python3
"""
Test client for the Bayesian Optimization API, demonstrating real-world usage.
This example optimizes a simulated chemistry reaction with multiple parameters
and includes constraints on the parameter space.
"""

import requests
import json
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from typing import Dict, List, Any, Tuple, Optional

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

# Function to create an optimization campaign with constraints
def create_optimization(optimizer_id: str, use_constraints: bool = True) -> bool:
    """Create a chemical reaction optimization with multiple parameters and constraints."""
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
    
    # Add constraints if enabled
    if use_constraints:
                    # Define constraints using BayBE's constraint system
        payload["constraints"] = [
            # Temperature pressure constraint using ContinuousLinearConstraint
            # This is encoded as: Temperature/120 + Pressure/5 <= 1.7
            {
                "type": "ContinuousLinear",
                "parameters": ["Temperature", "Pressure"],
                "operator": "<=",
                "coefficients": [1/120, 1/5],
                "rhs": 1.7
            },
            
            # Water-solvent temperature constraint
            # Using DiscreteExcludeConstraint with ThresholdCondition
            {
                "type": "DiscreteExclude",
                "parameters": ["Temperature", "Solvent"],
                "conditions": [
                    {
                        "type": "Threshold",
                        "threshold": 100,
                        "operator": ">="
                    },
                    {
                        "type": "SubSelection",
                        "selection": ["Water"]
                    }
                ],
                "combiner": "AND"
            },
            
            # Constraint: If temperature < 80, then time must be >= 60
            # Using DiscreteExcludeConstraint to exclude invalid combinations
            {
                "type": "DiscreteExclude",
                "parameters": ["Temperature", "Time"],
                "conditions": [
                    {
                        "type": "Threshold",
                        "threshold": 80,
                        "operator": "<"
                    },
                    {
                        "type": "Threshold",
                        "threshold": 60,
                        "operator": "<"
                    }
                ],
                "combiner": "AND"
            }
        ]

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
            
            # Print constraint information if available
            if use_constraints and "constraint_count" in result:
                print(f"\nApplied {result['constraint_count']} constraints to the search space")
            
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
def add_initial_measurements(optimizer_id: str, num_initial: int = 5, with_constraints: bool = True) -> List[Dict]:
    """Add initial measurements to bootstrap the Bayesian optimization."""
    print(f"\n=== Adding {num_initial} Initial Measurements ===\n")
    initial_experiments = []
    
    # Define some diverse initial points to explore the parameter space
    # Modified to respect constraints if enabled
    if with_constraints:
        initial_params = [
            {"Temperature": 70, "Pressure": 2.0, "Time": 60, "Catalyst": "B", "Solvent": "Ethanol"},
            {"Temperature": 90, "Pressure": 3.0, "Time": 90, "Catalyst": "C", "Solvent": "Acetone"},
            # Water with lower temperature to respect constraints
            {"Temperature": 50, "Pressure": 1.5, "Time": 75, "Catalyst": "A", "Solvent": "Water"},
            # Lower pressure with high temperature to respect constraints
            {"Temperature": 110, "Pressure": 2.5, "Time": 120, "Catalyst": "E", "Solvent": "THF"},
            {"Temperature": 80, "Pressure": 2.5, "Time": 75, "Catalyst": "D", "Solvent": "Methanol"}
        ]
    else:
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
        
        # Check if this point respects constraints (when constraints are enabled)
        if with_constraints:
            # Manual verification of constraints for demonstration
            constraints_ok = True
            
            # Check temperature/pressure constraint
            if temperature/120 + pressure/5 > 1.7:
                print(f"WARNING: Point violates temperature-pressure constraint")
                constraints_ok = False
                
            # Check water-temperature constraint  
            if solvent == "Water" and temperature >= 100:
                print(f"WARNING: Point violates water-temperature constraint")
                constraints_ok = False
                
            # Check temperature-time constraint
            if temperature < 80 and reaction_time < 60:
                print(f"WARNING: Point violates temperature-time constraint")
                constraints_ok = False
                
            if not constraints_ok:
                print(f"This point violates constraints but will be used for demonstration purposes.")
        
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
def run_optimization(optimizer_id: str, num_iterations: int = 20, initial_experiments: List[Dict] = None,
                     with_constraints: bool = True) -> List[Dict]:
    print(f"\n=== Running {num_iterations} Bayesian Optimization Iterations ===\n")
    all_experiments = initial_experiments or []
    constraint_violations = 0
    
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
        
        # Check if this recommendation respects constraints (for verification purposes)
        if with_constraints:
            constraints_ok = True
            
            # Check temperature/pressure constraint
            if temperature/120 + pressure/5 > 1.7:
                print(f"WARNING: Recommendation violates temperature-pressure constraint")
                constraints_ok = False
                
            # Check water-temperature constraint  
            if solvent == "Water" and temperature >= 100:
                print(f"WARNING: Recommendation violates water-temperature constraint")
                constraints_ok = False
                
            # Check temperature-time constraint
            if temperature < 80 and reaction_time < 60:
                print(f"WARNING: Recommendation violates temperature-time constraint")
                constraints_ok = False
                
            if not constraints_ok:
                constraint_violations += 1
        
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
    
    if with_constraints:
        print(f"\nConstraint violations detected in recommendations: {constraint_violations}/{num_iterations}")
        if constraint_violations > 0:
            print("Note: Violations might indicate issues with constraint implementation")
        else:
            print("Success! All recommendations satisfied the defined constraints")
    
    return all_experiments

# Function to plot the optimization results
def plot_results(experiments: List[Dict], with_constraints: bool = True):
    # Extract data for plotting
    iterations = [exp["iteration"] for exp in experiments]
    yields = [exp["yield"] for exp in experiments]
    
    # Calculate best yield at each iteration (cumulative max)
    best_yields = [max(yields[:i+1]) for i in range(len(yields))]
    
    # Create figure with three subplots if using constraints
    if with_constraints:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
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
    
    # Plot 3: Constraint visualization (if using constraints)
    if with_constraints:
        # Create a temperature-pressure plot to visualize the first constraint
        temps = [exp["temperature"] for exp in experiments]
        pressures = [exp["pressure"] for exp in experiments]
        
        # Create background grid for constraint visualization
        temp_grid = np.linspace(50, 120, 100)
        press_grid = np.linspace(1.0, 5.0, 100)
        T, P = np.meshgrid(temp_grid, press_grid)
        
        # Constraint 1: Temperature/120 + Pressure/5 <= 1.7
        Z = T/120 + P/5
        constraint_mask = (Z <= 1.7)
        
        # Plot the constraint boundary
        ax3.contourf(T, P, Z, levels=[0, 1.7, 3], alpha=0.3, 
                     colors=['lightgreen', 'lightcoral'])
        ax3.contour(T, P, Z, levels=[1.7], colors=['red'], linewidths=2)
        
        # Plot the data points
        solvent_colors = {'Water': 'blue', 'Methanol': 'green', 'Ethanol': 'purple', 
                          'Acetone': 'orange', 'THF': 'brown'}
        
        for i, exp in enumerate(experiments):
            color = solvent_colors[exp['solvent']]
            marker = 'o' if exp['temperature'] >= 80 or exp['time'] >= 60 else 'X'
            
            # Check if this point satisfies constraint 1
            constraint1_ok = (exp['temperature']/120 + exp['pressure']/5 <= 1.7)
            
            # Use different edge colors based on constraint satisfaction
            edgecolor = 'black' if constraint1_ok else 'red'
            linewidth = 1 if constraint1_ok else 2
            
            ax3.scatter(exp['temperature'], exp['pressure'], 
                       color=color, marker=marker, s=80, 
                       edgecolor=edgecolor, linewidth=linewidth,
                       alpha=0.7)
            
            # Add iteration number as text
            ax3.annotate(str(exp['iteration']), 
                        (exp['temperature'], exp['pressure']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
            
        # Add legend for solvents
        for solvent, color in solvent_colors.items():
            ax3.scatter([], [], color=color, label=solvent)
            
        # Add legend for constraint satisfaction
        ax3.scatter([], [], color='white', edgecolor='black', label='Constraints OK')
        ax3.scatter([], [], color='white', edgecolor='red', linewidth=2, label='Constraint violated')
        
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Pressure (bar)')
        ax3.set_title('Temperature-Pressure Space\nwith Constraint Visualization')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right', fontsize=8)
        
        # Add annotations explaining the constraint regions
        ax3.text(60, 4.5, "Valid Region\n(T/120 + P/5 <= 1.7)", 
                color='darkgreen', fontsize=9, ha='left')
        ax3.text(110, 4.5, "Invalid Region\n(T/120 + P/5 > 1.7)", 
                color='darkred', fontsize=9, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    filename = 'constrained_optimization_results.png' if with_constraints else 'optimization_results.png'
    plt.savefig(filename)
    print(f"Results plot saved as '{filename}'")
    plt.show()

# Function to run comparative experiments with and without constraints
def run_comparative_experiment():
    # Generate unique optimizer IDs
    base_id = int(time.time())
    optimizer_id_constrained = f"chemical-reaction-constrained-{base_id}"
    optimizer_id_unconstrained = f"chemical-reaction-unconstrained-{base_id}"
    
    # First run with constraints
    print("\n\n======= RUNNING OPTIMIZATION WITH CONSTRAINTS =======\n")
    print(f"Creating optimization with ID: {optimizer_id_constrained}")
    
    if create_optimization(optimizer_id_constrained, use_constraints=True):
        initial_constrained = add_initial_measurements(optimizer_id_constrained, num_initial=5, with_constraints=True)
        experiments_constrained = run_optimization(
            optimizer_id_constrained, 
            num_iterations=15,
            initial_experiments=initial_constrained,
            with_constraints=True
        )
        
        # Get the best result
        best_constrained = get_best_point(optimizer_id_constrained)
        
        # Then run without constraints
        print("\n\n======= RUNNING OPTIMIZATION WITHOUT CONSTRAINTS =======\n")
        print(f"Creating optimization with ID: {optimizer_id_unconstrained}")
        
        if create_optimization(optimizer_id_unconstrained, use_constraints=False):
            initial_unconstrained = add_initial_measurements(optimizer_id_unconstrained, num_initial=5, with_constraints=False)
            experiments_unconstrained = run_optimization(
                optimizer_id_unconstrained, 
                num_iterations=15,
                initial_experiments=initial_unconstrained,
                with_constraints=False
            )
            
            # Get the best result
            best_unconstrained = get_best_point(optimizer_id_unconstrained)
            
            # Compare results
            print("\n\n======= COMPARISON OF RESULTS =======\n")
            
            if "best_parameters" in best_constrained and "best_parameters" in best_unconstrained:
                print("Best results with constraints:")
                print(f"  Parameters: {best_constrained['best_parameters']}")
                print(f"  Yield: {best_constrained['best_value']:.2f}%")
                
                print("\nBest results without constraints:")
                print(f"  Parameters: {best_unconstrained['best_parameters']}")
                print(f"  Yield: {best_unconstrained['best_value']:.2f}%")
                
                # Check if unconstrained result violates constraints
                unconstrained_best = best_unconstrained['best_parameters']
                violations = []
                
                # Check temperature/pressure constraint
                temp = unconstrained_best['Temperature']
                press = unconstrained_best['Pressure']
                if temp/120 + press/5 > 1.7:
                    violations.append("temperature-pressure constraint")
                    
                # Check water-temperature constraint
                if unconstrained_best['Solvent'] == 'Water' and temp >= 100:
                    violations.append("water-temperature constraint")
                    
                # Check temperature-time constraint
                if temp < 80 and unconstrained_best['Time'] < 60:
                    violations.append("temperature-time constraint")
                
                if violations:
                    print(f"\nWarning: Unconstrained best result violates {', '.join(violations)}")
                    print("This demonstrates the importance of constraints for realistic optimizations.")
                else:
                    print("\nInterestingly, the unconstrained best result respects all constraints.")
                
                # Plot results
                print("\nPlotting constrained optimization results...")
                plot_results(experiments_constrained, with_constraints=True)
                
                print("\nPlotting unconstrained optimization results...")
                plot_results(experiments_unconstrained, with_constraints=False)
                
                return {
                    "constrained": {
                        "experiments": experiments_constrained,
                        "best": best_constrained
                    },
                    "unconstrained": {
                        "experiments": experiments_unconstrained,
                        "best": best_unconstrained
                    }
                }
            
    return None

# Main execution
if __name__ == "__main__":
    print("=== Bayesian Optimization API Test Client with Constraints ===")
    
    # Check if the API is healthy and using GPU
    using_gpu = test_health()
    print(f"\nAPI using GPU acceleration: {using_gpu}")
    
    # Ask if user wants to run comparison or single experiment
    run_comparison = input("\nRun comparison between constrained and unconstrained optimization? (y/n): ").lower().startswith('y')
    
    if run_comparison:
        results = run_comparative_experiment()
    else:
        # Generate a unique optimizer ID
        optimizer_id = f"chemical-reaction-{int(time.time())}"
        use_constraints = input("Use constraints? (y/n): ").lower().startswith('y')
        print(f"\nCreating optimization with ID: {optimizer_id}")
        
        # Create the optimization
        if create_optimization(optimizer_id, use_constraints=use_constraints):
            # Add initial measurements first (before requesting recommendations)
            initial_experiments = add_initial_measurements(optimizer_id, num_initial=5, with_constraints=use_constraints)
            
            # Then run the optimization loop
            all_experiments = run_optimization(
                optimizer_id, 
                num_iterations=15,
                initial_experiments=initial_experiments,
                with_constraints=use_constraints
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
                plot_results(all_experiments, with_constraints=use_constraints)
    
    print("\nTest client execution completed.")