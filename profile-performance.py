#!/usr/bin/env python3
"""
GPU Performance Profiling Tool for the Bayesian Optimization API

This script benchmarks the performance difference between CPU and GPU for 
different optimization scenarios. It measures the time taken to generate
recommendations as the complexity (number of parameters, batch size) increases.
"""

import requests
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import threading
import argparse
from datetime import datetime

try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
    GPU_AVAILABLE = True
except:
    try:
        import pynvml
        pynvml.nvmlInit()
        GPU_AVAILABLE = True
    except:
        GPU_AVAILABLE = False

# API configuration
API_URL = "http://localhost:8000"
API_KEY = "123456789"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Monitoring intervals
MONITOR_INTERVAL = 0.5  # seconds
gpu_metrics = []
cpu_metrics = []
memory_metrics = []
timestamp_metrics = []
monitoring = False

# Function to monitor system resources
def monitor_resources():
    global monitoring, gpu_metrics, cpu_metrics, memory_metrics, timestamp_metrics
    
    while monitoring:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics if available
        gpu_usage = 0
        gpu_memory = 0
        if GPU_AVAILABLE:
            try:
                # Try with nvidia_smi first
                try:
                    nvidia_smi.nvmlInit()
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                    util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory = mem_info.used / mem_info.total * 100
                except:
                    # Fall back to pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory = mem_info.used / mem_info.total * 100
            except Exception as e:
                pass
        
        # Record metrics
        timestamp = time.time()
        cpu_metrics.append(cpu_percent)
        gpu_metrics.append(gpu_usage)
        memory_metrics.append(memory_usage)
        timestamp_metrics.append(timestamp)
        
        # Print current metrics every 10 samples
        if len(cpu_metrics) % 10 == 0:
            print(f"Current - CPU: {cpu_percent}%, GPU: {gpu_usage}%, Memory: {memory_usage}%")
        
        time.sleep(MONITOR_INTERVAL)

# Function to ensure API is running and check GPU status
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return False, False
            
        data = response.json()
        using_gpu = data.get('using_gpu', False)
        
        print(f"API Status: {data.get('status', 'unknown')}")
        print(f"API Using GPU: {using_gpu}")
        
        if using_gpu and data.get('gpu_info'):
            print(f"GPU: {data['gpu_info'].get('name', 'unknown')}")
        
        return True, using_gpu
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False, False

# Function to create an optimization problem
def create_optimization(optimizer_id, num_parameters=5, num_values=10):
    """Create an optimization problem with the specified complexity"""
    
    # Generate parameters with complexity based on input args
    parameters = []
    
    # Add numerical parameters
    for i in range(num_parameters - 2):  # Reserve 2 slots for categorical params
        param_values = list(np.linspace(0, 100, num_values))
        parameters.append({
            "type": "NumericalDiscrete",
            "name": f"Parameter_{i+1}",
            "values": param_values,
            "tolerance": 0.1
        })
    
    # Add two categorical parameters
    categorical_values = [f"Cat_{j}" for j in range(min(num_values, 10))]
    parameters.append({
        "type": "CategoricalParameter",
        "name": "Categorical_1",
        "values": categorical_values,
        "encoding": "OHE"
    })
    
    parameters.append({
        "type": "CategoricalParameter",
        "name": "Categorical_2",
        "values": categorical_values[:5],  # Use fewer values for second categorical
        "encoding": "OHE"
    })
    
    # Create the optimization config
    payload = {
        "parameters": parameters,
        "target_config": {
            "name": "Target",
            "mode": "MAX"
        },
        "recommender_config": {
            "type": "TwoPhaseMetaRecommender",  # Changed to enable initial recommendations
            "initial_recommender": {
                "type": "FPSRecommender"
            },
            "recommender": {
                "type": "BotorchRecommender",
                "n_restarts": 10,
                "n_raw_samples": 64
            }
        }
    }
    
    try:
        response = requests.post(
            f"{API_URL}/optimization/{optimizer_id}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            print(f"Optimization '{optimizer_id}' created successfully")
            return True
        else:
            print(f"Error creating optimization: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

# Function to simulate a simple objective function
def objective_function(params):
    """Simulate a complex objective function"""
    result = 0
    for key, value in params.items():
        if isinstance(value, (int, float)):
            # For numerical parameters
            result += np.sin(value/10) * np.cos(value/20) + np.random.normal(0, 0.1)
        else:
            # For categorical parameters, use hash as a proxy for value
            result += (hash(value) % 100) / 100.0
    
    return 50 + result * 10  # Scale to reasonable range

# Function to add measurements to the optimization
def add_measurements(optimizer_id, num_samples=10):
    """Add initial measurements to the optimization"""
    
    # First get the parameters and create sample points
    response = requests.get(
        f"{API_URL}/optimization/{optimizer_id}/suggest?batch_size={num_samples}",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error getting suggestions: {response.text}")
        return False
    
    suggestions = response.json()["suggestions"]
    
    # Create measurements
    measurements = []
    for params in suggestions:
        target_value = objective_function(params)
        measurements.append({
            "parameters": params,
            "target_value": target_value
        })
    
    # Add measurements one by one
    for measurement in measurements:
        try:
            response = requests.post(
                f"{API_URL}/optimization/{optimizer_id}/measurement",
                headers=headers,
                json=measurement
            )
            
            if response.status_code != 200:
                print(f"Error adding measurement: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"Added {len(measurements)} measurements to optimization '{optimizer_id}'")
    return True

# Function to benchmark recommendation time
def benchmark_recommendation(optimizer_id, batch_size=1, repeat=3):
    """Benchmark the time it takes to generate recommendations"""
    
    times = []
    
    print(f"Benchmarking recommendation with batch_size={batch_size}...")
    
    for i in range(repeat):
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{API_URL}/optimization/{optimizer_id}/suggest?batch_size={batch_size}",
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"Error getting suggestion: {response.text}")
                continue
                
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            print(f"  Run {i+1}: {elapsed:.4f} seconds")
            
        except Exception as e:
            print(f"Error: {e}")
    
    return {
        "batch_size": batch_size,
        "mean_time": np.mean(times) if times else None,
        "std_time": np.std(times) if times else None,
        "min_time": np.min(times) if times else None,
        "max_time": np.max(times) if times else None,
        "times": times
    }

# Function to run a full benchmark with varying complexity
def run_benchmarks(using_gpu):
    """Run a series of benchmarks with different parameters"""
    
    results = []
    
    # Configuration for benchmarks
    parameter_counts = [3, 5, 8, 10, 15, 20]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    # Adjust based on whether GPU is available
    if using_gpu:
        # With GPU we can test more complex scenarios
        parameter_counts.extend([25, 30])
        batch_sizes.extend([64, 128])
    
    print("\n=== Starting Benchmarks ===\n")
    
    # 1. Benchmark by number of parameters
    print("\n--- Benchmarking by Parameter Count ---\n")
    for param_count in parameter_counts:
        optimizer_id = f"benchmark_params_{param_count}_{int(time.time())}"
        
        print(f"\nCreating optimization with {param_count} parameters...")
        if not create_optimization(optimizer_id, num_parameters=param_count):
            continue
            
        # Add some initial measurements
        if not add_measurements(optimizer_id, num_samples=10):
            continue
            
        # Start resource monitoring
        global monitoring, gpu_metrics, cpu_metrics, memory_metrics, timestamp_metrics
        monitoring = True
        gpu_metrics = []
        cpu_metrics = []
        memory_metrics = []
        timestamp_metrics = []
        
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Benchmark with a moderate batch size
        benchmark_result = benchmark_recommendation(optimizer_id, batch_size=8, repeat=5)
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join(timeout=1.0)
        
        # Add results
        result = {
            "type": "params",
            "params": param_count,
            "batch": 8,
            "using_gpu": using_gpu,
            "mean_time": benchmark_result["mean_time"],
            "std_time": benchmark_result["std_time"],
            "min_time": benchmark_result["min_time"],
            "max_time": benchmark_result["max_time"],
            "cpu_mean": np.mean(cpu_metrics) if cpu_metrics else None,
            "gpu_mean": np.mean(gpu_metrics) if gpu_metrics else None,
            "memory_mean": np.mean(memory_metrics) if memory_metrics else None
        }
        
        results.append(result)
        print(f"Result: {result}")
    
    # 2. Benchmark by batch size (fixed parameter count)
    print("\n--- Benchmarking by Batch Size ---\n")
    param_count = 10  # Use a moderate number of parameters
    
    optimizer_id = f"benchmark_batch_{param_count}_{int(time.time())}"
    
    print(f"\nCreating optimization with {param_count} parameters...")
    if create_optimization(optimizer_id, num_parameters=param_count):
        # Add some initial measurements
        if add_measurements(optimizer_id, num_samples=10):
            for batch_size in batch_sizes:
                # Start resource monitoring
                monitoring = True
                gpu_metrics = []
                cpu_metrics = []
                memory_metrics = []
                timestamp_metrics = []
                
                monitor_thread = threading.Thread(target=monitor_resources)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Benchmark with this batch size
                benchmark_result = benchmark_recommendation(optimizer_id, batch_size=batch_size, repeat=3)
                
                # Stop monitoring
                monitoring = False
                monitor_thread.join(timeout=1.0)
                
                # Add results
                result = {
                    "type": "batch",
                    "params": param_count,
                    "batch": batch_size,
                    "using_gpu": using_gpu,
                    "mean_time": benchmark_result["mean_time"],
                    "std_time": benchmark_result["std_time"],
                    "min_time": benchmark_result["min_time"],
                    "max_time": benchmark_result["max_time"],
                    "cpu_mean": np.mean(cpu_metrics) if cpu_metrics else None,
                    "gpu_mean": np.mean(gpu_metrics) if gpu_metrics else None,
                    "memory_mean": np.mean(memory_metrics) if memory_metrics else None
                }
                
                results.append(result)
                print(f"Result: {result}")
    
    return results

# Function to generate plots from benchmark results
def generate_plots(results, plot_filename="benchmark_results.png"):
    """Generate plots visualizing the benchmark results"""
    
    # Split results by type
    param_results = [r for r in results if r["type"] == "params"]
    batch_results = [r for r in results if r["type"] == "batch"]
    
    # Sort results
    param_results.sort(key=lambda x: x["params"])
    batch_results.sort(key=lambda x: x["batch"])
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Parameters vs Time
    if param_results:
        gpu_results = [r for r in param_results if r["using_gpu"]]
        cpu_results = [r for r in param_results if not r["using_gpu"]]
        
        ax = axs[0, 0]
        
        if gpu_results:
            ax.plot(
                [r["params"] for r in gpu_results],
                [r["mean_time"] for r in gpu_results],
                'o-', color='green', label='GPU'
            )
            
        if cpu_results:
            ax.plot(
                [r["params"] for r in cpu_results],
                [r["mean_time"] for r in cpu_results],
                'o-', color='blue', label='CPU'
            )
            
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Recommendation Time vs Parameter Count')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Plot 2: Batch Size vs Time
    if batch_results:
        gpu_results = [r for r in batch_results if r["using_gpu"]]
        cpu_results = [r for r in batch_results if not r["using_gpu"]]
        
        ax = axs[0, 1]
        
        if gpu_results:
            ax.plot(
                [r["batch"] for r in gpu_results],
                [r["mean_time"] for r in gpu_results],
                'o-', color='green', label='GPU'
            )
            
        if cpu_results:
            ax.plot(
                [r["batch"] for r in cpu_results],
                [r["mean_time"] for r in cpu_results],
                'o-', color='blue', label='CPU'
            )
            
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Recommendation Time vs Batch Size')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Plot 3: Parameters vs Speedup
    if param_results:
        # Match GPU and CPU results by parameter count
        gpu_by_params = {r["params"]: r for r in param_results if r["using_gpu"]}
        cpu_by_params = {r["params"]: r for r in param_results if not r["using_gpu"]}
        
        # Find common parameter counts
        common_params = sorted(set(gpu_by_params.keys()) & set(cpu_by_params.keys()))
        
        if common_params:
            ax = axs[1, 0]
            
            # Calculate speedup
            speedups = [
                cpu_by_params[p]["mean_time"] / gpu_by_params[p]["mean_time"]
                for p in common_params
                if cpu_by_params[p]["mean_time"] is not None and gpu_by_params[p]["mean_time"] is not None
            ]
            
            ax.bar(common_params, speedups, color='orange', alpha=0.7)
            ax.axhline(y=1.0, color='red', linestyle='--')
            
            ax.set_xlabel('Number of Parameters')
            ax.set_ylabel('Speedup (CPU time / GPU time)')
            ax.set_title('GPU Speedup vs Parameter Count')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Batch Size vs Speedup
    if batch_results:
        # Match GPU and CPU results by batch size
        gpu_by_batch = {r["batch"]: r for r in batch_results if r["using_gpu"]}
        cpu_by_batch = {r["batch"]: r for r in batch_results if not r["using_gpu"]}
        
        # Find common batch sizes
        common_batch = sorted(set(gpu_by_batch.keys()) & set(cpu_by_batch.keys()))
        
        if common_batch:
            ax = axs[1, 1]
            
            # Calculate speedup
            speedups = [
                cpu_by_batch[b]["mean_time"] / gpu_by_batch[b]["mean_time"]
                for b in common_batch
                if cpu_by_batch[b]["mean_time"] is not None and gpu_by_batch[b]["mean_time"] is not None
            ]
            
            ax.bar(common_batch, speedups, color='orange', alpha=0.7)
            ax.axhline(y=1.0, color='red', linestyle='--')
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Speedup (CPU time / GPU time)')
            ax.set_title('GPU Speedup vs Batch Size')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Benchmark plots saved to {plot_filename}")
    plt.show()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark the Bayesian Optimization API')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (ignore GPU)')
    parser.add_argument('--output', default='benchmark_results.csv', help='Output CSV file for results')
    args = parser.parse_args()
    
    print(f"=== Bayesian Optimization API Performance Profiling ===")
    print(f"Starting time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API status and GPU availability
    api_running, using_gpu = check_api_status()
    
    if not api_running:
        print("Error: API is not running. Please start the API before running benchmarks.")
        exit(1)
    
    # Override GPU detection if --cpu flag is used
    if args.cpu:
        using_gpu = False
        print("Forcing CPU mode due to --cpu flag")
    
    print(f"\nStarting benchmarks with {'GPU' if using_gpu else 'CPU'} mode...")
    all_results = run_benchmarks(using_gpu)
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
        
        # Generate plots
        generate_plots(all_results)
    
    print(f"\nBenchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")