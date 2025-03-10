import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from baybe import Campaign
from baybe.searchspace import SearchSpace
from baybe.constraints.base import Constraint
from baybe.constraints import (
    ContinuousLinearConstraint,
    ContinuousCardinalityConstraint,
    DiscreteCardinalityConstraint,
    DiscreteCustomConstraint,
    DiscreteDependenciesConstraint,
    DiscreteExcludeConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
    SubSelectionCondition,
    ThresholdCondition,
    validate_constraints
)
from baybe.objectives import SingleTargetObjective, DesirabilityObjective
from baybe.parameters import (
    NumericalDiscreteParameter, 
    NumericalContinuousParameter,
    CategoricalParameter,
    SubstanceParameter
)
from baybe.targets import NumericalTarget
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender, 
    TwoPhaseMetaRecommender
)


class BayesianOptimizationBackend:
    """A Bayesian Optimization backend using BayBE with GPU acceleration."""
    
    def __init__(self, storage_path: str = "./data", use_gpu: bool = True):
        """Initialize the Bayesian optimization backend.
        
        Args:
            storage_path: Directory to store optimization data.
            use_gpu: Whether to use GPU acceleration if available.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_campaigns = {}
        
        # Set up GPU if requested and available
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if use_gpu and not torch.cuda.is_available():
            print("WARNING: GPU requested but not available. Using CPU instead.")
        else:
            print(f"Using device: {self.device}")
            
        # Configure PyTorch to use the selected device
        if self.device.type == "cuda":
            # Set default tensor type to cuda
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # Additional GPU optimizations
            torch.backends.cudnn.benchmark = True
            
            # Log GPU information
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_cached(0) / 1e6:.2f} MB")

    def create_optimization(
        self, 
        optimizer_id: str,
        parameters: List[Dict[str, Any]],
        target_config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        recommender_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Create a new optimization process.
        
        Args:
            optimizer_id: Unique identifier for this optimization.
            parameters: List of parameter configurations.
            target_config: Configuration for the optimization target.
            constraints: Optional list of constraint configurations.
            recommender_config: Optional recommender configuration.
            
        Returns:
            Dictionary with status information.
        """
        # 1. Parameter Parsing and Instantiation
        parsed_parameters = []
        
        for param_config in parameters:
            param_config = param_config.copy()  # Create a copy to avoid modifying the original
            param_type = param_config.pop("type")
            
            if param_type == "NumericalDiscrete":
                param = NumericalDiscreteParameter(**param_config)
            elif param_type == "NumericalContinuous":
                param = NumericalContinuousParameter(**param_config)
            elif param_type in ["Categorical", "CategoricalParameter"]:  # Handle both formats
                param = CategoricalParameter(**param_config)
            elif param_type in ["Substance", "SubstanceParameter"]:  # Handle both formats
                param = SubstanceParameter(**param_config)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            parsed_parameters.append(param)
        
        # 2. Constraints Processing
        parsed_constraints = []
        if constraints:
            for constraint_config in constraints:
                constraint_config = constraint_config.copy()  # Create a copy to avoid modifying the original
                constraint_type = constraint_config.pop("type")
                
                # Map constraint type to the appropriate class
                constraint_class = self._get_constraint_class(constraint_type)
                
                # Process condition if needed
                if "condition" in constraint_config:
                    condition_config = constraint_config.pop("condition")
                    condition_type = condition_config.pop("type", None)
                    
                    if condition_type == "Threshold":
                        constraint_config["condition"] = ThresholdCondition(**condition_config)
                    elif condition_type == "SubSelection":
                        constraint_config["condition"] = SubSelectionCondition(**condition_config)
                
                # Process conditions if needed (for constraints that take multiple conditions)
                if "conditions" in constraint_config and isinstance(constraint_config["conditions"], list):
                    processed_conditions = []
                    for cond in constraint_config["conditions"]:
                        cond_copy = cond.copy()  # Create a copy to avoid modifying the original
                        cond_type = cond_copy.pop("type", None)
                        if cond_type == "Threshold":
                            processed_conditions.append(ThresholdCondition(**cond_copy))
                        elif cond_type == "SubSelection":
                            processed_conditions.append(SubSelectionCondition(**cond_copy))
                    constraint_config["conditions"] = processed_conditions
                    
                # Remove any non-essential fields that might confuse the constraint constructor
                if "description" in constraint_config:
                    constraint_config.pop("description")
                
                # Create the constraint instance
                try:
                    constraint = constraint_class(**constraint_config)
                    parsed_constraints.append(constraint)
                except Exception as e:
                    raise ValueError(f"Error creating constraint of type {constraint_type}: {str(e)}")
        
        # 3. Validate constraints against parameters
        if parsed_constraints:
            try:
                validate_constraints(parsed_constraints, parsed_parameters)
            except Exception as e:
                return {"status": "error", "message": f"Constraint validation failed: {str(e)}"}
        
        # 4. SearchSpace Creation with Structure and Constraints
        try:
            searchspace = SearchSpace.from_product(
                parameters=parsed_parameters,
                constraints=parsed_constraints if parsed_constraints else None
            )
        except Exception as e:
            return {"status": "error", "message": f"Error creating search space: {str(e)}"}
            
        # Parse target
        target_name = target_config.get("name", "Target")
        target_mode = target_config.get("mode", "MAX")
        target = NumericalTarget(name=target_name, mode=target_mode)
        
        if target_config.get("bounds"):
            target.bounds = target_config["bounds"]
            
        objective = SingleTargetObjective(target=target)
        
        # Setup recommender
        if recommender_config is None:
            # Create a recommender configuration optimized for GPU when available
            if self.device.type == "cuda":
                # Use more parallel processing for GPU
                recommender = TwoPhaseMetaRecommender(
                    initial_recommender=FPSRecommender(),
                    recommender=BotorchRecommender(
                        n_restarts=20,  # Increased for GPU
                        n_raw_samples=128  # Increased for GPU
                    )
                )
            else:
                # Default configuration for CPU
                recommender = TwoPhaseMetaRecommender(
                    initial_recommender=FPSRecommender(),
                    recommender=BotorchRecommender()
                )
        else:
            # Parse recommender config (simplified)
            recommender_config = recommender_config.copy()  # Create a copy
            recommender_type = recommender_config.pop("type", "TwoPhaseMetaRecommender")
            if recommender_type == "TwoPhaseMetaRecommender":
                if self.device.type == "cuda":
                    # Optimize for GPU
                    recommender = TwoPhaseMetaRecommender(
                        initial_recommender=FPSRecommender(),
                        recommender=BotorchRecommender(
                            n_restarts=20,  # Increased for GPU
                            n_raw_samples=128,  # Increased for GPU
                            **recommender_config
                        )
                    )
                else:
                    recommender = TwoPhaseMetaRecommender(
                        initial_recommender=FPSRecommender(),
                        recommender=BotorchRecommender(**recommender_config)
                    )
            elif recommender_type == "BotorchRecommender":
                if self.device.type == "cuda":
                    # Update config with GPU optimizations
                    if "n_restarts" not in recommender_config:
                        recommender_config["n_restarts"] = 20
                    if "n_raw_samples" not in recommender_config:
                        recommender_config["n_raw_samples"] = 128
                recommender = BotorchRecommender(**recommender_config)
            else:
                raise ValueError(f"Unknown recommender type: {recommender_type}")
        
        # Create campaign
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=recommender
        )
        
        # Store campaign
        self.active_campaigns[optimizer_id] = campaign
        self.save_campaign(optimizer_id)
        
        return {
            "status": "success", 
            "message": "Optimization created",
            "constraint_count": len(parsed_constraints) if constraints else 0
        }
    
    def _get_constraint_class(self, constraint_type: str) -> type:
        """Map constraint type string to the appropriate constraint class.
        
        Args:
            constraint_type: String identifier for the constraint type.
            
        Returns:
            The corresponding constraint class.
            
        Raises:
            ValueError: If the constraint type is unknown.
        """
        constraint_map = {
            "ContinuousLinear": ContinuousLinearConstraint,
            "Linear": ContinuousLinearConstraint,  # Alias for backward compatibility
            "ContinuousCardinality": ContinuousCardinalityConstraint,
            "DiscreteCardinality": DiscreteCardinalityConstraint, 
            "DiscreteCustom": DiscreteCustomConstraint,
            "DiscreteDependencies": DiscreteDependenciesConstraint,
            "DiscreteExclude": DiscreteExcludeConstraint,
            "DiscreteLinkedParameters": DiscreteLinkedParametersConstraint,
            "DiscreteNoLabelDuplicates": DiscreteNoLabelDuplicatesConstraint,
            "DiscretePermutationInvariance": DiscretePermutationInvarianceConstraint,
            "DiscreteProduct": DiscreteProductConstraint,
            "DiscreteSum": DiscreteSumConstraint,
            
            # Map the generic "Nonlinear" type to DiscreteExclude for backward compatibility
            "Nonlinear": DiscreteExcludeConstraint
        }
        
        if constraint_type not in constraint_map:
            raise ValueError(f"Unknown constraint type: {constraint_type}. Available types: {list(constraint_map.keys())}")
        
        return constraint_map[constraint_type]
    
    def suggest_next_point(
        self, 
        optimizer_id: str, 
        batch_size: int = 1,
        pending_experiments: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Get the next suggested point(s) to evaluate.
        
        Args:
            optimizer_id: Identifier for the optimization.
            batch_size: Number of suggestions to return.
            pending_experiments: Optional dataframe of pending experiments.
            
        Returns:
            Dictionary with suggested points.
        """
        if optimizer_id not in self.active_campaigns:
            return self._load_or_error(optimizer_id)
        
        campaign = self.active_campaigns[optimizer_id]
        try:
            # Use GPU-accelerated optimization if available
            with torch.cuda.amp.autocast(enabled=self.device.type=="cuda"):
                suggestions = campaign.recommend(
                    batch_size=batch_size, 
                    pending_experiments=pending_experiments
                )
            
            return {
                "status": "success",
                "suggestions": suggestions.to_dict(orient="records")
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def add_measurement(
        self, 
        optimizer_id: str, 
        parameters: Dict[str, Any], 
        target_value: float
    ) -> Dict[str, str]:
        """Add a new measurement to the optimizer.
        
        Args:
            optimizer_id: Identifier for the optimization.
            parameters: Parameter values for this measurement.
            target_value: Measured target value.
            
        Returns:
            Dictionary with status information.
        """
        if optimizer_id not in self.active_campaigns:
            return self._load_or_error(optimizer_id)
        
        campaign = self.active_campaigns[optimizer_id]
        
        # Create measurement dataframe
        df = pd.DataFrame([parameters])
        target_name = campaign.targets[0].name
        df[target_name] = target_value
        
        try:
            campaign.add_measurements(df)
            self.save_campaign(optimizer_id)
            
            return {"status": "success", "message": "Measurement added"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def add_multiple_measurements(
        self, 
        optimizer_id: str, 
        measurements: pd.DataFrame
    ) -> Dict[str, str]:
        """Add multiple measurements at once.
        
        Args:
            optimizer_id: Identifier for the optimization.
            measurements: Dataframe with parameter values and target values.
            
        Returns:
            Dictionary with status information.
        """
        if optimizer_id not in self.active_campaigns:
            return self._load_or_error(optimizer_id)
        
        campaign = self.active_campaigns[optimizer_id]
        
        try:
            campaign.add_measurements(measurements)
            self.save_campaign(optimizer_id)
            
            return {"status": "success", "message": f"{len(measurements)} measurements added"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_best_point(self, optimizer_id: str) -> Dict[str, Any]:
        """Get the current best point.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with best point information.
        """
        if optimizer_id not in self.active_campaigns:
            return self._load_or_error(optimizer_id)
        
        campaign = self.active_campaigns[optimizer_id]
        
        if campaign.measurements.empty:
            return {
                "status": "success",
                "message": "No measurements yet"
            }
        
        # Find the best measurement
        target = campaign.targets[0]
        target_name = target.name
        measurements = campaign.measurements
        
        if target.mode == "MAX":
            best_idx = measurements[target_name].idxmax()
        else:  # MIN
            best_idx = measurements[target_name].idxmin()
            
        best_row = measurements.loc[best_idx]
        
        return {
            "status": "success",
            "best_parameters": best_row.drop(target_name).to_dict(),
            "best_value": best_row[target_name]
        }
    
    def save_campaign(self, optimizer_id: str) -> None:
        """Save the campaign state to disk.
        
        Args:
            optimizer_id: Identifier for the optimization.
        """
        campaign = self.active_campaigns[optimizer_id]
        file_path = self.storage_path / f"{optimizer_id}.json"
        
        # If using GPU, temporarily move to CPU for serialization to avoid CUDA tensor issues
        if self.device.type == "cuda":
            # This is a defensive approach since BayBE should handle this internally
            # but we're adding it just to be sure
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        # Convert campaign to JSON
        config_json = campaign.to_json()
        
        with open(file_path, 'w') as f:
            f.write(config_json)
    
    def load_campaign(self, optimizer_id: str) -> Dict[str, str]:
        """Load a campaign from disk.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with status information.
        """
        file_path = self.storage_path / f"{optimizer_id}.json"
        
        if not file_path.exists():
            return {
                "status": "error", 
                "message": f"No saved campaign found for ID: {optimizer_id}"
            }
        
        try:
            with open(file_path, 'r') as f:
                config_json = f.read()
            
            campaign = Campaign.from_config(config_json)
            self.active_campaigns[optimizer_id] = campaign
            
            # If using GPU, ensure any models are on the right device
            if self.device.type == "cuda":
                # Models should automatically use the default tensor type we set
                # This is a safeguard for any potential edge cases
                torch.cuda.empty_cache()
            
            return {"status": "success", "message": "Campaign loaded"}
        except Exception as e:
            return {"status": "error", "message": f"Error loading campaign: {str(e)}"}
    
    def _load_or_error(self, optimizer_id: str) -> Dict[str, str]:
        """Attempt to load a campaign or return an error.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with status information.
        """
        result = self.load_campaign(optimizer_id)
        if result["status"] == "success":
            return result
        else:
            return {"status": "error", "message": f"Optimizer not found: {optimizer_id}"}