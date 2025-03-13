# backend.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd
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
    validate_constraints,
    ContinuousLinearConstraint
)
from baybe.objectives import (
    SingleTargetObjective, 
    DesirabilityObjective, 
    # ParetoObjective - unreleased yet
)
from baybe.parameters import (
    NumericalDiscreteParameter, 
    NumericalContinuousParameter,
    CategoricalParameter,
    SubstanceParameter
)
from baybe.targets import NumericalTarget, BinaryTarget
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender
)
from baybe.surrogates import (
    GaussianProcessSurrogate,
    BayesianLinearSurrogate,
    MeanPredictionSurrogate,
    NGBoostSurrogate,
    RandomForestSurrogate
)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BayesianOptimizationBackend:
    """A Bayesian Optimization backend using BayBE with GPU acceleration."""
    
    def __init__(self, storage_path: str = "./data", use_gpu: bool = True):
        """Initialize the Bayesian optimization backend.
        
        Args:
            storage_path: Directory to store optimization data.
            use_gpu: Whether to use GPU acceleration if available.
            
        Returns:
            None
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_campaigns = {}
        
        # Set up GPU if requested and available
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Using CPU instead.")
        else:
            logger.info(f"Using device: {self.device}")
            
        # Configure PyTorch to use the selected device
        if self.device.type == "cuda":
            # Set default tensor type to cuda
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # Additional GPU optimizations
            torch.backends.cudnn.benchmark = True
            
            # Log GPU information
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
            logger.info(f"Memory cached: {torch.cuda.memory_cached(0) / 1e6:.2f} MB")
    
    def _parse_parameter(self, param_config: Dict[str, Any]) -> Union[
            NumericalDiscreteParameter, 
            NumericalContinuousParameter,
            CategoricalParameter,
            SubstanceParameter
        ]:
        """Parse a parameter configuration into a Parameter object.
        
        Args:
            param_config: Parameter configuration dictionary.
            
        Returns:
            The corresponding parameter object.
            
        Raises:
            ValueError: If the parameter type is unknown.
        """
        param_config = param_config.copy()  # Create a copy to avoid modifying the original
        param_type = param_config.pop("type")
        
        if param_type == "NumericalDiscrete":
            return NumericalDiscreteParameter(**param_config)
        elif param_type == "NumericalContinuous":
            return NumericalContinuousParameter(**param_config)
        elif param_type in ["Categorical", "CategoricalParameter"]:  # Handle both formats
            return CategoricalParameter(**param_config)
        elif param_type in ["Substance", "SubstanceParameter"]:  # Handle both formats
            return SubstanceParameter(**param_config)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _process_condition(self, condition_config: Dict[str, Any]) -> Union[ThresholdCondition, SubSelectionCondition]:
        """Process a condition configuration into a Condition object.
        
        Args:
            condition_config: Condition configuration dictionary.
            
        Returns:
            The corresponding condition object.
            
        Raises:
            ValueError: If the condition type is unknown.
        """
        condition_config = condition_config.copy()
        condition_type = condition_config.pop("type", None)
        
        if condition_type == "Threshold":
            return ThresholdCondition(**condition_config)
        elif condition_type == "SubSelection":
            return SubSelectionCondition(**condition_config)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")
    
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
            # "Linear": ContinuousLinearConstraint,  # Alias for backward compatibility
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
            "ContinuousLinearConstraint": ContinuousLinearConstraint
            # "LinearEquality": LinearEqualityConstraint,
            # "LinearInequality": LinearInequalityConstraint,
            # "Nonlinear": NonlinearConstraint,  # Map the generic "Nonlinear" type
        }
        
        if constraint_type not in constraint_map:
            raise ValueError(f"Unknown constraint type: {constraint_type}. Available types: {list(constraint_map.keys())}")
        
        return constraint_map[constraint_type]
    
    def _parse_constraint(self, 
                          constraint_config: Dict[str, Any], 
                          parameter_map: Optional[Dict[str, Any]] = None
                         ) -> Constraint:
        """Parse a constraint configuration into a Constraint object.
        
        Args:
            constraint_config: Constraint configuration dictionary.
            parameter_map: Optional dictionary mapping parameter names to parameter objects.
            
        Returns:
            The corresponding constraint object.
            
        Raises:
            ValueError: If the constraint type is unknown or configuration is invalid.
        """
        constraint_config = constraint_config.copy()
        constraint_type = constraint_config.pop("type")
        
        # Get the constraint class
        constraint_class = self._get_constraint_class(constraint_type)
        
        # Process parameters for constraints that reference parameters by name
        if parameter_map is not None and "parameters" in constraint_config:
            param_names = constraint_config.pop("parameters")
            constraint_params = [parameter_map[name] for name in param_names]
            constraint_config["parameters"] = constraint_params
        
        # Process condition if needed
        if "condition" in constraint_config and isinstance(constraint_config["condition"], dict):
            constraint_config["condition"] = self._process_condition(constraint_config["condition"])
        
        # Process conditions if needed (for constraints that take multiple conditions)
        if "conditions" in constraint_config and isinstance(constraint_config["conditions"], list):
            processed_conditions = []
            for cond in constraint_config["conditions"]:
                processed_conditions.append(self._process_condition(cond))
            constraint_config["conditions"] = processed_conditions
        
        # Special handling for DiscreteCustomConstraint
        if constraint_type == "DiscreteCustom" and "constraint_func" in constraint_config:
            constraint_func_str = constraint_config.pop("constraint_func")
            
            # Security check for unsafe code
            unsafe_keywords = ['__import__', 'eval', 'exec', 'open', 'os.', 'sys.', 'subprocess']
            if any(keyword in constraint_func_str for keyword in unsafe_keywords):
                raise ValueError("Unsafe keywords detected in constraint function")
            
            # Create a function from the string
            try:
                exec_locals = {}
                exec(f"def constraint_func(params):\n{constraint_func_str}", {}, exec_locals)
                constraint_func = exec_locals["constraint_func"]
                constraint_config["constraint_func"] = constraint_func
            except Exception as e:
                raise ValueError(f"Error in constraint function: {str(e)}")
        
        # Remove any non-essential fields that might confuse the constraint constructor
        if "description" in constraint_config:
            constraint_config.pop("description")
        
        # Create the constraint instance
        try:
            return constraint_class(**constraint_config)
        except Exception as e:
            raise ValueError(f"Error creating constraint of type {constraint_type}: {str(e)}")
    
    def _parse_target(self, target_config: Dict[str, Any]) -> Union[NumericalTarget, BinaryTarget]:
        """Parse a target configuration into a Target object.
        
        Args:
            target_config: Target configuration dictionary.
            
        Returns:
            The corresponding target object.
        """
        target_name = target_config.get("name", "Target")
        target_mode = target_config.get("mode", "MAX")
        
        # Check if it's a binary target
        if target_config.get("type") == "Binary":
            target = BinaryTarget(name=target_name, mode=target_mode)
        else:
            target = NumericalTarget(name=target_name, mode=target_mode)
            
        if target_config.get("bounds"):
            target.bounds = target_config["bounds"]
            
        return target
    
    def _configure_surrogate_model(self, config: Dict[str, Any]) -> Any:
        """Configure a surrogate model based on provided configuration.
        
        Args:
            config: Dictionary with surrogate model configuration.
                Must contain 'type' key specifying the model type.
                
        Returns:
            Configured surrogate model instance.
            
        Raises:
            ValueError: If the surrogate model type is unknown.
        """
        config = config.copy()
        model_type = config.pop("type", "GaussianProcess")
        
        if model_type == "GaussianProcess":
            return GaussianProcessSurrogate(**config)
        elif model_type == "BayesianLinear":
            return BayesianLinearSurrogate(**config)
        elif model_type == "MeanPrediction":
            return MeanPredictionSurrogate(**config)
        elif model_type == "NGBoost":
            return NGBoostSurrogate(**config)
        elif model_type == "RandomForest":
            return RandomForestSurrogate(**config)
        else:
            raise ValueError(f"Unknown surrogate model type: {model_type}")
    
    def _configure_acquisition_function(self, 
                                       config: Optional[Dict[str, Any]], 
                                       objective_type: str
                                      ) -> Dict[str, Any]:
        """Configure an acquisition function based on provided configuration and objective type.
        
        Args:
            config: Dictionary with acquisition function configuration or None.
            objective_type: Type of the objective (SingleTarget, Desirability, (or Pareto - when available)).
            
        Returns:
            Dictionary with acquisition function configuration.
        """
        if config is None:
            config = {}
        else:
            config = config.copy()
            
        acq_type = config.pop("type", None)
        
        # If no acquisition function is specified, choose appropriate default based on objective type
        if acq_type is None:
            if objective_type == "Pareto":
                return {"type": "qLogNoisyExpectedHypervolumeImprovement"}
            else:
                return {"type": "qLogExpectedImprovement"}  # Good default for most cases
        
        # Validate that the acquisition function is compatible with the objective type
        if objective_type == "Pareto" and acq_type not in ["qLogNoisyExpectedHypervolumeImprovement"]:
            raise ValueError(f"Acquisition function {acq_type} is not compatible with Pareto objectives")
        
        return {"type": acq_type, **config}
    
    def _configure_recommender(self, 
                              config: Optional[Dict[str, Any]], 
                              objective_type: str,
                              surrogate_config: Optional[Dict[str, Any]] = None,
                              acquisition_config: Optional[Dict[str, Any]] = None
                             ) -> Any:
        """Configure a recommender based on provided configuration.
        
        Args:
            config: Dictionary with recommender configuration or None.
            objective_type: Type of the objective (SingleTarget, Desirability, or Pareto).
            surrogate_config: Optional surrogate model configuration.
            acquisition_config: Optional acquisition function configuration.
                
        Returns:
            Configured recommender instance.
            
        Raises:
            ValueError: If the recommender type is unknown.
        """
        # Default to TwoPhaseMetaRecommender if not specified
        if config is None:
            recommender_type = "TwoPhaseMetaRecommender"
            config = {}
        else:
            config = config.copy()
            recommender_type = config.pop("type", "TwoPhaseMetaRecommender")
        
        # Configure surrogate model if provided
        surrogate_model = None
        if surrogate_config is not None:
            surrogate_model = self._configure_surrogate_model(surrogate_config)
        
        # Configure acquisition function if provided
        acquisition_function = None
        if acquisition_config is not None:
            acquisition_function = self._configure_acquisition_function(acquisition_config, objective_type)
        
        # Set GPU-optimized defaults if using CUDA
        cuda_optimizations = {}
        if self.device.type == "cuda":
            cuda_optimizations = {
                "n_restarts": 20,
                "n_raw_samples": 128
            }
        
        if recommender_type == "TwoPhaseMetaRecommender":
            # Configure initial recommender (default to FPSRecommender)
            initial_config = config.pop("initial_recommender", {"type": "FPSRecommender"})
            if isinstance(initial_config, dict):
                initial_type = initial_config.pop("type", "FPSRecommender")
                
                if initial_type == "FPSRecommender":
                    initial_recommender = FPSRecommender(**initial_config)
                elif initial_type == "RandomRecommender":
                    initial_recommender = RandomRecommender(**initial_config)
                else:
                    raise ValueError(f"Unknown initial recommender type: {initial_type}")
            else:
                initial_recommender = initial_config
            
            # Configure main recommender (default to BotorchRecommender)
            main_config = config.pop("recommender", {})
            if isinstance(main_config, dict):
                main_type = main_config.pop("type", "BotorchRecommender")
                
                if main_type == "BotorchRecommender":
                    recommender_kwargs = {**cuda_optimizations, **main_config}
                    
                    # Add surrogate model and acquisition function if provided
                    if surrogate_model is not None:
                        recommender_kwargs["surrogate_model"] = surrogate_model
                    if acquisition_function is not None:
                        recommender_kwargs["acquisition_function"] = acquisition_function
                        
                    main_recommender = BotorchRecommender(**recommender_kwargs)
                else:
                    raise ValueError(f"Unknown main recommender type: {main_type}")
            else:
                main_recommender = main_config
            
            # Create meta-recommender
            switch_after = config.pop("switch_after", 1)
            remain_switched = config.pop("remain_switched", True)
            
            return TwoPhaseMetaRecommender(
                initial_recommender=initial_recommender,
                recommender=main_recommender,
                switch_after=switch_after,
                remain_switched=remain_switched,
                **config
            )
        
        elif recommender_type == "BotorchRecommender":
            recommender_kwargs = {**cuda_optimizations, **config}
            
            # Add surrogate model and acquisition function if provided
            if surrogate_model is not None:
                recommender_kwargs["surrogate_model"] = surrogate_model
            if acquisition_function is not None:
                recommender_kwargs["acquisition_function"] = acquisition_function
                
            return BotorchRecommender(**recommender_kwargs)
        
        elif recommender_type == "FPSRecommender":
            return FPSRecommender(**config)
        
        elif recommender_type == "RandomRecommender":
            return RandomRecommender(**config)
        
        else:
            raise ValueError(f"Unknown recommender type: {recommender_type}")
    
    def create_optimization(
        self, 
        optimizer_id: str,
        parameters: List[Dict[str, Any]],
        target_config: Union[Dict[str, Any], List[Dict[str, Any]]],
        constraints: Optional[List[Dict[str, Any]]] = None,
        recommender_config: Optional[Dict[str, Any]] = None,
        objective_type: str = "SingleTarget",
        surrogate_config: Optional[Dict[str, Any]] = None,
        acquisition_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Create a new optimization process.
        
        Args:
            optimizer_id: Unique identifier for this optimization.
            parameters: List of parameter configurations.
            target_config: Configuration for the optimization target(s).
                Can be a single dict for single-objective or a list of dicts for multi-objective.
            constraints: Optional list of constraint configurations.
            recommender_config: Optional recommender configuration.
            objective_type: Type of objective to use. One of:
                - "SingleTarget": Single objective optimization
                - "Desirability": Multiple objectives combined with weights
                - "Pareto": True multi-objective optimization with Pareto front # when available
            surrogate_config: Optional surrogate model configuration.
            acquisition_config: Optional acquisition function configuration.
            
        Returns:
            Dictionary with status information.
            
        Raises:
            ValueError: If the parameters, constraints, or objective configuration is invalid.
        """
        try:
            # 1. Parameter Parsing and Instantiation
            parsed_parameters = []
            parameter_map = {}  # For referencing parameters in constraints
            
            for param_config in parameters:
                try:
                    param = self._parse_parameter(param_config)
                    parsed_parameters.append(param)
                    parameter_map[param_config.get("name")] = param
                except Exception as e:
                    return {"status": "error", "message": f"Error parsing parameter: {str(e)}"}
            
            # 2. Constraints Processing
            parsed_constraints = []
            if constraints:
                for constraint_config in constraints:
                    try:
                        constraint = self._parse_constraint(constraint_config, parameter_map)
                        parsed_constraints.append(constraint)
                    except Exception as e:
                        return {"status": "error", "message": f"Error parsing constraint: {str(e)}"}
            
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
            
            # 5. Parse target(s) and create objective
            if objective_type == "SingleTarget":
                # For backward compatibility with single target case
                if isinstance(target_config, list):
                    # If a list is provided but SingleTarget is specified, use the first target
                    target_config = target_config[0]
                
                try:
                    target = self._parse_target(target_config)
                    objective = SingleTargetObjective(target=target)
                except Exception as e:
                    return {"status": "error", "message": f"Error creating target: {str(e)}"}
                
            elif objective_type == "Desirability":
                # For desirability, we need multiple targets
                if not isinstance(target_config, list):
                    return {"status": "error", "message": "For Desirability objective, target_config must be a list of targets"}
                
                try:
                    targets = []
                    weights = []
                    
                    for target_conf in target_config:
                        target = self._parse_target(target_conf)
                        targets.append(target)
                        weights.append(target_conf.get("weight", 1.0))
                    
                    # Create desirability objective with the specified targets and weights
                    objective = DesirabilityObjective(
                        targets=targets, 
                        weights=weights,
                        mean_type=target_config[0].get("mean_type", "arithmetic")  # Get from first target or default
                    )
                except Exception as e:
                    return {"status": "error", "message": f"Error creating desirability objective: {str(e)}"}
                
            # elif objective_type == "Pareto":
            #     # For Pareto optimization
            #     if not isinstance(target_config, list):
            #         return {"status": "error", "message": "For Pareto objective, target_config must be a list of targets"}
                
            #     try:
            #         targets = []
                    
            #         for target_conf in target_config:
            #             target = self._parse_target(target_conf)
            #             targets.append(target)
                    
            #         # Create Pareto objective with the specified targets
            #         objective = ParetoObjective(targets=targets)
            #     except Exception as e:
            #         return {"status": "error", "message": f"Error creating Pareto objective: {str(e)}"}
            # else:
            #     return {"status": "error", "message": f"Unknown objective type: {objective_type}"}
            
            # 6. Configure recommender
            try:
                recommender = self._configure_recommender(
                    recommender_config,
                    objective_type,
                    surrogate_config,
                    acquisition_config
                )
            except Exception as e:
                return {"status": "error", "message": f"Error configuring recommender: {str(e)}"}
            
            # 7. Create campaign
            try:
                campaign = Campaign(
                    searchspace=searchspace,
                    objective=objective,
                    recommender=recommender
                )
            except Exception as e:
                return {"status": "error", "message": f"Error creating campaign: {str(e)}"}
            
            # 8. Store campaign
            self.active_campaigns[optimizer_id] = campaign
            save_result = self.save_campaign(optimizer_id)
            if save_result.get("status") == "error":
                return save_result
            
            return {
                "status": "success", 
                "message": "Optimization created",
                "optimizer_id": optimizer_id,
                "parameter_count": len(parsed_parameters),
                "constraint_count": len(parsed_constraints) if constraints else 0
            }
        
        except Exception as e:
            logger.error(f"Error creating optimization: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}
    
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
            load_result = self._load_or_error(optimizer_id)
            if load_result.get("status") == "error":
                return load_result
        
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
                "suggestions": suggestions.to_dict(orient="records"),
                "batch_size": batch_size
            }
        except Exception as e:
            logger.error(f"Error suggesting next point: {str(e)}", exc_info=True)
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
            load_result = self._load_or_error(optimizer_id)
            if load_result.get("status") == "error":
                return load_result
        
        campaign = self.active_campaigns[optimizer_id]
        
        # Create measurement dataframe
        df = pd.DataFrame([parameters])
        target_name = campaign.targets[0].name
        df[target_name] = target_value
        
        try:
            campaign.add_measurements(df)
            save_result = self.save_campaign(optimizer_id)
            if save_result.get("status") == "error":
                return save_result
            
            return {"status": "success", "message": "Measurement added"}
        except Exception as e:
            logger.error(f"Error adding measurement: {str(e)}", exc_info=True)
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
            load_result = self._load_or_error(optimizer_id)
            if load_result.get("status") == "error":
                return load_result
        
        campaign = self.active_campaigns[optimizer_id]
        
        try:
            campaign.add_measurements(measurements)
            save_result = self.save_campaign(optimizer_id)
            if save_result.get("status") == "error":
                return save_result
            
            return {"status": "success", "message": f"{len(measurements)} measurements added"}
        except Exception as e:
            logger.error(f"Error adding multiple measurements: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def get_best_point(self, optimizer_id: str) -> Dict[str, Any]:
        """Get the current best point.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with best point information.
        """
        if optimizer_id not in self.active_campaigns:
            load_result = self._load_or_error(optimizer_id)
            if load_result.get("status") == "error":
                return load_result
        
        campaign = self.active_campaigns[optimizer_id]
        
        if campaign.measurements.empty:
            return {
                "status": "success",
                "message": "No measurements yet"
            }
        
        try:
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
                "best_value": float(best_row[target_name]),
                "total_measurements": len(measurements)
            }
        except Exception as e:
            logger.error(f"Error getting best point: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def save_campaign(self, optimizer_id: str) -> Dict[str, str]:
        """Save the campaign state to disk with comprehensive error handling.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with status information and error details if applicable.
        """
        try:
            campaign = self.active_campaigns[optimizer_id]
            file_path = self.storage_path / f"{optimizer_id}.json"
            
            # If using GPU, temporarily move to CPU for serialization to avoid CUDA tensor issues
            if self.device.type == "cuda":
                # This is a defensive approach since BayBE should handle this internally
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Could not empty CUDA cache: {str(e)}")
            
            # Convert campaign to JSON
            try:
                config_json = campaign.to_json()
            except Exception as e:
                return {"status": "error", "message": f"Error serializing campaign: {str(e)}"}
            
            # Write to file with error handling
            try:
                with open(file_path, 'w') as f:
                    f.write(config_json)
                return {"status": "success", "message": "Campaign saved successfully"}
            except Exception as e:
                return {"status": "error", "message": f"Error writing campaign to disk: {str(e)}"}
        except Exception as e:
            logger.error(f"Error saving campaign: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}
    
    # New methods added below
    
    def _load_or_error(self, optimizer_id: str) -> Dict[str, str]:
        """Load a campaign or return an error message.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with status information.
        """
        # First, check if the campaign exists on disk
        file_path = self.storage_path / f"{optimizer_id}.json"
        if not file_path.exists():
            return {"status": "error", "message": f"Optimizer {optimizer_id} not found"}
        
        # Try to load the campaign
        try:
            load_result = self.load_campaign(optimizer_id)
            if load_result["status"] == "error":
                return load_result
            return {"status": "success", "message": f"Optimizer {optimizer_id} loaded successfully"}
        except Exception as e:
            logger.error(f"Error loading optimizer: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error loading optimizer: {str(e)}"}

    def load_campaign(self, optimizer_id: str) -> Dict[str, str]:
        """Load a campaign from disk.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with status information.
        """
        try:
            file_path = self.storage_path / f"{optimizer_id}.json"
            
            if not file_path.exists():
                return {"status": "error", "message": f"Optimizer not found: {optimizer_id}"}
            
            # Read the JSON file
            with open(file_path, 'r') as f:
                config_json = f.read()
            
            # Create campaign from JSON
            from baybe import Campaign
            campaign = Campaign.from_json(config_json)
            
            # Store in active campaigns
            self.active_campaigns[optimizer_id] = campaign
            
            return {"status": "success", "message": f"Optimizer {optimizer_id} loaded successfully"}
        except Exception as e:
            logger.error(f"Error loading campaign: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error loading optimizer: {str(e)}"}

    def get_measurement_history(self, optimizer_id: str) -> Dict[str, Any]:
        """Get the measurement history for an optimization.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with measurement history.
        """
        if optimizer_id not in self.active_campaigns:
            load_result = self._load_or_error(optimizer_id)
            if load_result.get("status") == "error":
                return {"status": "error", "message": load_result["message"]}
        
        campaign = self.active_campaigns[optimizer_id]
        
        if campaign.measurements.empty:
            return {
                "status": "success",
                "message": "No measurements yet",
                "measurements": []
            }
        
        return {
            "status": "success",
            "measurements": campaign.measurements.to_dict(orient="records")
        }

    def get_campaign_info(self, optimizer_id: str) -> Dict[str, Any]:
        """Get information about a campaign.
        
        Args:
            optimizer_id: Identifier for the optimization.
            
        Returns:
            Dictionary with campaign information.
        """
        if optimizer_id not in self.active_campaigns:
            load_result = self._load_or_error(optimizer_id)
            if load_result.get("status") == "error":
                return {"status": "error", "message": load_result["message"]}
        
        campaign = self.active_campaigns[optimizer_id]
        
        # Collect campaign information
        info = {
            "parameters": [],
            "target": {},
            "measurements_count": len(campaign.measurements) if hasattr(campaign, "measurements") else 0
        }
        
        # Add parameter information
        for param in campaign.searchspace.parameters:
            param_info = {
                "name": param.name,
                "type": param.__class__.__name__
            }
            
            if hasattr(param, "values") and param.values is not None:
                param_info["values"] = param.values.tolist() if hasattr(param.values, 'tolist') else param.values
            elif hasattr(param, "bounds") and param.bounds is not None:
                param_info["bounds"] = [param.bounds.lower, param.bounds.upper]
                
            info["parameters"].append(param_info)
        
        # Add target information
        target = campaign.targets[0]
        info["target"] = {
            "name": target.name,
            "mode": target.mode
        }
        
        if hasattr(target, "bounds") and target.bounds is not None:
            info["target"]["bounds"] = [target.bounds.lower, target.bounds.upper]
        
        return {
            "status": "success",
            "info": info
        }

    def list_optimizations(self) -> Dict[str, Any]:
        """List all available optimizations.
        
        Returns:
            Dictionary with list of optimizations.
        """
        try:
            # Get all JSON files in the storage directory
            json_files = list(self.storage_path.glob("*.json"))
            
            # Extract optimizer IDs from filenames
            optimizer_ids = [file.stem for file in json_files]
            
            # Get information for each optimizer
            optimizers = []
            
            for optimizer_id in optimizer_ids:
                # Try to load basic information without loading the entire campaign
                try:
                    file_path = self.storage_path / f"{optimizer_id}.json"
                    
                    with open(file_path, 'r') as f:
                        # Read first few lines to extract basic info
                        first_lines = "".join([f.readline() for _ in range(20)])
                    
                    # Check if it's a valid campaign file
                    if "searchspace" in first_lines and "objective" in first_lines:
                        optimizers.append({
                            "id": optimizer_id,
                            "file_path": str(file_path)
                        })
                except Exception as e:
                    # Skip files that can't be parsed
                    logger.warning(f"Could not parse file {file_path}: {str(e)}")
                    pass
            
            return {
                "status": "success",
                "optimizers": optimizers
            }
        except Exception as e:
            logger.error(f"Error listing optimizations: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error listing optimizations: {str(e)}"}