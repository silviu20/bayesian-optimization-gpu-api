"""
Script to test if the GPU is properly configured for the Bayesian optimization API.
This can be run separately to verify GPU functionality before starting the main API.
"""

import torch
import sys

def test_gpu():
    print("===== GPU Configuration Test =====")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Create a simple tensor and move it to GPU
        print("Testing tensor operations on GPU...")
        x = torch.randn(1000, 1000)
        try:
            x_gpu = x.cuda()
            y_gpu = x_gpu @ x_gpu.T  # Matrix multiplication
            print("GPU tensor operations successful!")
            
            # Check compute capability
            device_props = torch.cuda.get_device_properties(0)
            print(f"Compute capability: {device_props.major}.{device_props.minor}")
            print(f"Total memory: {device_props.total_memory / 1e9:.2f} GB")
            
            return True
        except Exception as e:
            print(f"Error during GPU testing: {e}")
            return False
    else:
        print("No GPU detected. The API will run in CPU-only mode.")
        print("If you believe this is an error, check your CUDA installation and GPU drivers.")
        return False

def test_baybe_gpu():
    """Test BayBE package with GPU support"""
    try:
        from baybe import Campaign
        from baybe.searchspace import SearchSpace
        from baybe.objectives import SingleTargetObjective
        from baybe.parameters import NumericalDiscreteParameter
        from baybe.targets import NumericalTarget
        from baybe.recommenders import BotorchRecommender
        
        print("\n===== BayBE GPU Test =====")
        
        # Create a simple optimization problem
        parameter = NumericalDiscreteParameter(
            name="x", 
            values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        searchspace = SearchSpace.from_product([parameter])
        target = NumericalTarget(name="y", mode="MAX")
        objective = SingleTargetObjective(target=target)
        
        # Create the campaign with a BotorchRecommender
        recommender = BotorchRecommender()
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=recommender
        )
        
        # Get initial recommendations
        recommendations = campaign.recommend(batch_size=2)
        print(f"Initial recommendations:\n{recommendations}")
        
        # Simulate a measurement (simple quadratic function)
        measurements = recommendations.copy()
        measurements["y"] = measurements["x"].apply(lambda x: -(x - 0.7)**2 + 1.0)
        print(f"Measurements:\n{measurements}")
        
        # Add the measurements to the campaign
        campaign.add_measurements(measurements)
        
        # Get new recommendations
        new_recommendations = campaign.recommend(batch_size=2)
        print(f"New recommendations:\n{new_recommendations}")
        
        print("BayBE GPU test completed successfully!")
        return True
    except Exception as e:
        print(f"Error during BayBE testing: {e}")
        return False

if __name__ == "__main__":
    gpu_available = test_gpu()
    if gpu_available:
        baybe_success = test_baybe_gpu()
        if baybe_success:
            print("\nAll tests passed. The system is properly configured for GPU acceleration.")
            sys.exit(0)
        else:
            print("\nBayBE GPU test failed. Please check your configuration.")
            sys.exit(1)
    else:
        print("\nGPU test failed. The system will run in CPU-only mode.")
        sys.exit(1)