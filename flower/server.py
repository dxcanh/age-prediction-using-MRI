import flwr as fl
from flwr.common import (
    FitRes, Metrics, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy # Keep this for FedCrossStrategy inheritance
from flwr.server.strategy import FedAvg, FedProx # Import built-in strategies

import torch
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import os
import wandb
import logging

# Assuming these are in your project structure
from tsan_model.ScaleDense import ScaleDense
from central_server_logic.fedcross_logic import fedcross_aggregation, generate_global_model_pytorch
from central_server_logic.fedbn import fedbn_aggregation, generate_global_model_fedbn
from utils import state_dict_to_ndarrays, ndarrays_to_state_dict, get_parameters_from_model

logger = logging.getLogger("FlowerServer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cpu")

# --- Environment Configurations ---
USE_GENDER_CONFIG = os.environ.get("USE_GENDER", "True").lower() == "true"
EPOCHS_PER_ROUND = int(os.environ.get("EPOCHS_PER_ROUND", 3))
FL_STRATEGY_CHOICE = os.environ.get("FL_STRATEGY", "fedavg").lower() # fedavg, fedprox, fedcross
FL_PROXIMAL_MU = float(os.environ.get("FL_PROXIMAL_MU", 0.1)) # For FedProx

FEDCROSS_CONFIG = {
    "alpha": float(os.environ.get("FL_ALPHA", 0.9)),
    "collab_strategy": os.environ.get("FL_COLLAB_STRATEGY", "in_order"),
    "dynamic_alpha": os.environ.get("FL_DYNAMIC_ALPHA", "True").lower() == "true",
    "min_clients_per_round": int(os.environ.get("FL_MIN_CLIENTS", 2)) # Used by all strategies for consistency
}

WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_SERVER", "fl_tsan_project_server")
WANDB_RUN_GROUP_NAME = os.environ.get("WANDB_RUN_GROUP_SERVER", "federated_tsan_server")


class FedCrossStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        model_template: Optional[torch.nn.Module] = None, # Added model_template
        wandb_project_name: Optional[str] = None,
        wandb_run_group_name: Optional[str] = "fedcross_group"
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        
        if model_template is None:
            logger.error("FedCrossStrategy requires a model_template to be provided.")
            # Potentially raise an error or handle this state
        self.model_template = model_template.to(DEVICE) if model_template else None
        self.wandb_project_name = wandb_project_name
        self.wandb_run_group_name = wandb_run_group_name
        
        if self.wandb_project_name:
            try:
                wandb.init(
                    project=self.wandb_project_name,
                    group=self.wandb_run_group_name, # Use the group name passed or a default
                    name=os.environ.get("WANDB_RUN_NAME_SERVER", f"fedcross_server_strategy_run_{np.random.randint(1000, 9999)}"),
                    config={
                        "strategy": "FedCrossStrategy",
                        "min_fit_clients": self.min_fit_clients,
                        "fraction_fit": self.fraction_fit,
                        **FEDCROSS_CONFIG # Log specific FedCross params
                    },
                    reinit=True
                )
                logger.info(f"WandB initialized for FedCrossStrategy on project {self.wandb_project_name}")
            except Exception as e:
                logger.error(f"WandB initialization failed for FedCrossStrategy: {e}")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        logger.info("FedCrossStrategy: Initializing parameters")
        if self.initial_parameters is not None:
            if wandb.run and self.model_template: # Log model architecture if WandB is active for this strategy
                wandb.config.update({"model_architecture": str(self.model_template)})
            return self.initial_parameters
        logger.warning("FedCrossStrategy: Initial parameters were not provided at construction.")
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        config = {"server_round": server_round, "epochs_per_round": EPOCHS_PER_ROUND}
        fit_ins = fl.common.FitIns(parameters, config)
        
        num_clients_needed_for_fedcross = self.min_fit_clients # This is K for FedCross
        num_currently_available = client_manager.num_available()
        logger.info(f"Round {server_round} (FedCross): Num available clients = {num_currently_available}. Need {self.min_available_clients} to be available, and will try to sample {num_clients_needed_for_fedcross} for fit.")

        if num_currently_available < self.min_available_clients:
            logger.warning(
                f"Round {server_round} (FedCross): Not enough clients available ({num_currently_available}) "
                f"to meet overall min_available_clients ({self.min_available_clients}). "
                f"Server will likely wait or skip round based on main loop."
            )
            return []
        
        clients = client_manager.sample(
            num_clients=num_clients_needed_for_fedcross,
            min_num_clients=num_clients_needed_for_fedcross # For FedCross, we need exactly K clients
        )
        
        if len(clients) < num_clients_needed_for_fedcross:
            logger.warning(
                f"Round {server_round} (FedCross): Could not sample the required K={num_clients_needed_for_fedcross} clients "
                f"for FedCross fit (got {len(clients)}). min_available_clients is {self.min_available_clients}. "
                f"Consider adjusting client startup or min_fit_clients."
            )
            return []
        
        logger.info(f"Round {server_round} (FedCross): Successfully sampled {len(clients)} clients for FedCross fit (K={num_clients_needed_for_fedcross}).")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            logger.warning(f"Round {server_round} (FedCross): No fit results to aggregate. Failures: {len(failures)}")
            if wandb.run: # Log if WandB is active for this strategy
                wandb.log({"round": server_round, "fit_results_received": 0, "fit_failures": len(failures)}, commit=False if server_round == 0 else True)
            # Return current parameters if it's the first round and initial params exist, else None
            current_params = self.initial_parameters if server_round == 1 and self.initial_parameters else None
            return current_params, {}

        num_successful_results = len(results)
        logger.info(f"Round {server_round} (FedCross): Aggregating {num_successful_results} successful fit results.")

        client_updates_for_fedcross = []
        if not self.model_template:
            logger.error(f"Round {server_round} (FedCross): model_template is None in aggregate_fit. Cannot process results.")
            return None, {}

        for i, (_, fit_res) in enumerate(results):
            client_params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_state_dict = ndarrays_to_state_dict(client_params_ndarrays, self.model_template)
            client_updates_for_fedcross.append({
                'middleware_id': i, # Or use client.cid if available and needed by fedcross_aggregation
                'model': client_state_dict,
                'dataset_size': fit_res.num_examples # This is num_examples from client
            })
        
        current_k_for_fedcross = len(client_updates_for_fedcross)
        if current_k_for_fedcross == 0: # Should be caught by 'if not results' earlier, but good practice
            logger.warning(f"Round {server_round} (FedCross): No client updates to process for FedCross after parsing results.")
            return None, {}

        fedcross_internal_config = FEDCROSS_CONFIG.copy()
        fedcross_internal_config["total_rounds"] = int(os.environ.get("FL_TOTAL_ROUNDS", 50)) # FedCross might need total_rounds
        
        logger.info(f"Round {server_round} (FedCross): Calling fedcross_aggregation with {current_k_for_fedcross} client updates.")
        try:
            new_middleware_state_dicts = fedcross_aggregation(
                client_updates=client_updates_for_fedcross,
                K=current_k_for_fedcross, # The actual number of clients that participated
                round_num=server_round -1, # FedCross logic might be 0-indexed for rounds
                config=fedcross_internal_config,
                device=DEVICE
            )
        except Exception as e:
            logger.error(f"Round {server_round} (FedCross): Error during fedcross_aggregation: {e}", exc_info=True)
            if wandb.run: wandb.log({"round": server_round, "fedcross_aggregation_error": 1}, commit=False if server_round == 0 else True)
            return None, {} # Or return previous parameters if available

        if not new_middleware_state_dicts or len(new_middleware_state_dicts) != current_k_for_fedcross:
            logger.error(f"Round {server_round} (FedCross): fedcross_aggregation did not return expected models. Expected {current_k_for_fedcross}, got {len(new_middleware_state_dicts) if new_middleware_state_dicts else 0}.")
            if wandb.run: wandb.log({"round": server_round, "fedcross_model_mismatch": 1}, commit=False if server_round == 0 else True)
            return None, {}
            
        # Generate the single global model from the (potentially personalized) middleware models
        global_model_state_dict = generate_global_model_pytorch(
            new_middleware_state_dicts, # These are the updated models from FedCross for each client
            device=DEVICE
        )

        if not global_model_state_dict:
            logger.error(f"Round {server_round} (FedCross): Failed to generate global model from FedCross outputs.")
            if wandb.run: wandb.log({"round": server_round, "global_model_generation_failure": 1}, commit=False if server_round == 0 else True)
            return None, {}
            
        aggregated_parameters_ndarrays = state_dict_to_ndarrays(global_model_state_dict)
        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
        
        logger.info(f"Round {server_round} (FedCross): Aggregation complete.")
        
        metrics_aggregated = {} # Populate with any metrics FedCross might return
        if wandb.run: # Log if WandB is active for this strategy
            log_data = {"round": server_round, "fit_results_received": num_successful_results, "fit_failures": len(failures)}
            # Add any specific metrics from FedCross if available
            # e.g. log_data.update(new_middleware_state_dicts.get("metrics", {}))
            log_data.update(metrics_aggregated)
            wandb.log(log_data, commit=False if server_round == 0 else True)

        return aggregated_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        if self.fraction_evaluate == 0.0: # No evaluation
            return []
        
        # Sample clients for evaluation
        num_currently_available = client_manager.num_available()
        if num_currently_available < self.min_available_clients: # Check against overall min_available_clients
            logger.warning(f"Round {server_round} (FedCross): Not enough available clients ({num_currently_available}) for evaluation. min_available_clients is {self.min_available_clients}. Skipping evaluation.")
            return []

        # Determine number to sample based on min_evaluate_clients or fraction_evaluate
        # For simplicity, using min_evaluate_clients directly if fraction_evaluate is not 0.
        # Flower's default strategies handle fraction_evaluate logic internally if not overridden.
        num_to_sample_eval = self.min_evaluate_clients
        if num_to_sample_eval == 0 and num_currently_available > 0: # if min_eval is 0, try to eval on all available
            num_to_sample_eval = num_currently_available 
        elif num_to_sample_eval == 0 and num_currently_available == 0:
            logger.warning(f"Round {server_round} (FedCross): No clients available for evaluation, and min_evaluate_clients is 0.")
            return []


        clients = client_manager.sample(
            num_clients=num_to_sample_eval, 
            min_num_clients=self.min_evaluate_clients # Ensure at least this many are sampled if available
        )
        if not clients and self.min_evaluate_clients > 0: # Only warn if we expected clients
             logger.warning(f"Round {server_round} (FedCross): No clients could be sampled for evaluation despite {num_currently_available} available. min_evaluate_clients is {self.min_evaluate_clients}")
             return []
        
        logger.info(f"Round {server_round} (FedCross): Sampled {len(clients)} for evaluation.")
        config = {"server_round": server_round} # Send server_round to client for its own potential logging
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            logger.warning(f"Round {server_round} (FedCross): No evaluation results. Failures: {len(failures)}")
            if wandb.run: # Log if WandB is active
                wandb.log({"round": server_round, "eval_results_received": 0, "eval_failures": len(failures)}, commit=True)
            return None, {}

        num_examples_total = sum([evaluate_res.num_examples for _, evaluate_res in results])
        if num_examples_total == 0:
            logger.warning(f"Round {server_round} (FedCross): Total number of examples in evaluation results is 0. Cannot compute weighted average.")
            if wandb.run:
                wandb.log({"round": server_round, "eval_total_examples": 0}, commit=True)
            return None, {}


        # Standard weighted averaging for loss
        loss_aggregated = sum(
            [evaluate_res.num_examples * evaluate_res.loss for _, evaluate_res in results]
        ) / num_examples_total
        
        # Aggregate other metrics, e.g., MAE
        # Assuming 'mae' is in metrics dictionary from client
        maes = [res.metrics["mae"] * res.num_examples for _, res in results if "mae" in res.metrics]
        aggregated_mae = sum(maes) / num_examples_total if maes else float('inf') # Avoid division by zero if no MAE reported

        metrics_dict = {"loss": loss_aggregated, "mae": aggregated_mae}
        logger.info(f"Round {server_round} (FedCross): Evaluation aggregated. Loss: {loss_aggregated:.4f}, MAE: {aggregated_mae:.4f}")
        
        if wandb.run: # Log if WandB is active
            wandb.log({"round": server_round, "global_val_loss": loss_aggregated, "global_val_mae": aggregated_mae, "eval_results_received": len(results)}, commit=True) # Commit evaluation metrics immediately
        
        return loss_aggregated, metrics_dict

    def evaluate( # Server-side evaluation (optional, often unused if federated evaluation is comprehensive)
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        logger.info(f"Round {server_round} (FedCross): Server-side 'evaluate' method called by Flower. "
                    "This strategy relies on federated evaluation via 'aggregate_evaluate'. "
                    "No server-side dataset evaluation performed here.")
        
        if wandb.run: # Log if WandB is active
            wandb.log({"round": server_round, "server_evaluate_noop": 1}, commit=True) # Commit immediately
        return None # Returning None indicates no server-side evaluation
    
class FedBNStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        model_template: Optional[torch.nn.Module] = None,
        wandb_project_name: Optional[str] = None,
        wandb_run_group_name: Optional[str] = "fedbn_group"
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        
        if model_template is None:
            logger.error("FedBNStrategy requires a model_template to be provided.")
        self.model_template = model_template.to(DEVICE) if model_template else None
        self.wandb_project_name = wandb_project_name
        self.wandb_run_group_name = wandb_run_group_name
        
        if self.wandb_project_name:
            try:
                wandb.init(
                    project=self.wandb_project_name,
                    group=self.wandb_run_group_name,
                    name=os.environ.get("WANDB_RUN_NAME_SERVER", f"fedbn_server_strategy_run_{np.random.randint(1000, 9999)}"),
                    config={
                        "strategy": "FedBNStrategy",
                        "min_fit_clients": self.min_fit_clients,
                        "fraction_fit": self.fraction_fit,
                        "epochs_per_round": EPOCHS_PER_ROUND,
                        "use_gender": USE_GENDER_CONFIG
                    },
                    reinit=True
                )
                logger.info(f"WandB initialized for FedBNStrategy on project {self.wandb_project_name}")
            except Exception as e:
                logger.error(f"WandB initialization failed for FedBNStrategy: {e}")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        logger.info("FedBNStrategy: Initializing parameters")
        if self.initial_parameters is not None:
            if wandb.run and self.model_template:
                wandb.config.update({"model_architecture": str(self.model_template)})
            return self.initial_parameters
        logger.warning("FedBNStrategy: Initial parameters were not provided at construction.")
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        config = {"server_round": server_round, "epochs_per_round": EPOCHS_PER_ROUND}
        fit_ins = fl.common.FitIns(parameters, config)
        
        num_clients_needed = self.min_fit_clients
        num_currently_available = client_manager.num_available()
        logger.info(f"Round {server_round} (FedBN): Num available clients = {num_currently_available}. Need {self.min_available_clients} to be available, and will try to sample {num_clients_needed} for fit.")

        if num_currently_available < self.min_available_clients:
            logger.warning(
                f"Round {server_round} (FedBN): Not enough clients available ({num_currently_available}) "
                f"to meet overall min_available_clients ({self.min_available_clients}). "
                f"Server will likely wait or skip round based on main loop."
            )
            return []
        
        clients = client_manager.sample(
            num_clients=num_clients_needed,
            min_num_clients=num_clients_needed
        )
        
        if len(clients) < num_clients_needed:
            logger.warning(
                f"Round {server_round} (FedBN): Could not sample the required {num_clients_needed} clients "
                f"for FedBN fit (got {len(clients)}). min_available_clients is {self.min_available_clients}. "
                f"Consider adjusting client startup or min_fit_clients."
            )
            return []
        
        logger.info(f"Round {server_round} (FedBN): Successfully sampled {len(clients)} clients for FedBN fit.")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            logger.warning(f"Round {server_round} (FedBN): No fit results to aggregate. Failures: {len(failures)}")
            if wandb.run:
                wandb.log({"round": server_round, "fit_results_received": 0, "fit_failures": len(failures)}, commit=False if server_round == 0 else True)
            current_params = self.initial_parameters if server_round == 1 and self.initial_parameters else None
            return current_params, {}

        num_successful_results = len(results)
        logger.info(f"Round {server_round} (FedBN): Aggregating {num_successful_results} successful fit results.")

        client_updates_for_fedbn = []
        if not self.model_template:
            logger.error(f"Round {server_round} (FedBN): model_template is None in aggregate_fit. Cannot process results.")
            return None, {}

        for i, (_, fit_res) in enumerate(results):
            client_params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_state_dict = ndarrays_to_state_dict(client_params_ndarrays, self.model_template)
            client_updates_for_fedbn.append({
                'middleware_id': i,
                'model': client_state_dict,
                'dataset_size': fit_res.num_examples
            })
        
        current_k_for_fedbn = len(client_updates_for_fedbn)
        if current_k_for_fedbn == 0:
            logger.warning(f"Round {server_round} (FedBN): No client updates to process for FedBN after parsing results.")
            return None, {}

        fedbn_internal_config = {
            "total_rounds": int(os.environ.get("FL_TOTAL_ROUNDS", 50))
        }
        
        logger.info(f"Round {server_round} (FedBN): Calling fedbn_aggregation with {current_k_for_fedbn} client updates.")
        try:
            new_middleware_state_dicts = fedbn_aggregation(
                client_updates=client_updates_for_fedbn,
                K=current_k_for_fedbn,
                round_num=server_round - 1,  # FedBN logic might be 0-indexed for rounds
                config=fedbn_internal_config,
                device=DEVICE
            )
        except Exception as e:
            logger.error(f"Round {server_round} (FedBN): Error during fedbn_aggregation: {e}", exc_info=True)
            if wandb.run: 
                wandb.log({"round": server_round, "fedbn_aggregation_error": 1}, commit=False if server_round == 0 else True)
            return None, {}

        if not new_middleware_state_dicts or len(new_middleware_state_dicts) != current_k_for_fedbn:
            logger.error(f"Round {server_round} (FedBN): fedbn_aggregation did not return expected models. Expected {current_k_for_fedbn}, got {len(new_middleware_state_dicts) if new_middleware_state_dicts else 0}.")
            if wandb.run: 
                wandb.log({"round": server_round, "fedbn_model_mismatch": 1}, commit=False if server_round == 0 else True)
            return None, {}
            
        # Generate the single global model from the personalized models
        global_model_state_dict = generate_global_model_fedbn(
            new_middleware_state_dicts,
            device=DEVICE
        )

        if not global_model_state_dict:
            logger.error(f"Round {server_round} (FedBN): Failed to generate global model from FedBN outputs.")
            if wandb.run: 
                wandb.log({"round": server_round, "global_model_generation_failure": 1}, commit=False if server_round == 0 else True)
            return None, {}
            
        aggregated_parameters_ndarrays = state_dict_to_ndarrays(global_model_state_dict)
        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
        
        logger.info(f"Round {server_round} (FedBN): Aggregation complete.")
        
        metrics_aggregated = {}
        if wandb.run:
            log_data = {"round": server_round, "fit_results_received": num_successful_results, "fit_failures": len(failures)}
            log_data.update(metrics_aggregated)
            wandb.log(log_data, commit=False if server_round == 0 else True)

        return aggregated_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        
        num_currently_available = client_manager.num_available()
        if num_currently_available < self.min_available_clients:
            logger.warning(f"Round {server_round} (FedBN): Not enough available clients ({num_currently_available}) for evaluation. min_available_clients is {self.min_available_clients}. Skipping evaluation.")
            return []

        num_to_sample_eval = self.min_evaluate_clients
        if num_to_sample_eval == 0 and num_currently_available > 0:
            num_to_sample_eval = num_currently_available 
        elif num_to_sample_eval == 0 and num_currently_available == 0:
            logger.warning(f"Round {server_round} (FedBN): No clients available for evaluation, and min_evaluate_clients is 0.")
            return []

        clients = client_manager.sample(
            num_clients=num_to_sample_eval, 
            min_num_clients=self.min_evaluate_clients
        )
        if not clients and self.min_evaluate_clients > 0:
             logger.warning(f"Round {server_round} (FedBN): No clients could be sampled for evaluation despite {num_currently_available} available. min_evaluate_clients is {self.min_evaluate_clients}")
             return []
        
        logger.info(f"Round {server_round} (FedBN): Sampled {len(clients)} for evaluation.")
        config = {"server_round": server_round}
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            logger.warning(f"Round {server_round} (FedBN): No evaluation results. Failures: {len(failures)}")
            if wandb.run:
                wandb.log({"round": server_round, "eval_results_received": 0, "eval_failures": len(failures)}, commit=True)
            return None, {}

        num_examples_total = sum([evaluate_res.num_examples for _, evaluate_res in results])
        if num_examples_total == 0:
            logger.warning(f"Round {server_round} (FedBN): Total number of examples in evaluation results is 0. Cannot compute weighted average.")
            if wandb.run:
                wandb.log({"round": server_round, "eval_total_examples": 0}, commit=True)
            return None, {}

        # Standard weighted averaging for loss
        loss_aggregated = sum(
            [evaluate_res.num_examples * evaluate_res.loss for _, evaluate_res in results]
        ) / num_examples_total
        
        # Aggregate other metrics, e.g., MAE
        maes = [res.metrics["mae"] * res.num_examples for _, res in results if "mae" in res.metrics]
        aggregated_mae = sum(maes) / num_examples_total if maes else float('inf')

        metrics_dict = {"loss": loss_aggregated, "mae": aggregated_mae}
        logger.info(f"Round {server_round} (FedBN): Evaluation aggregated. Loss: {loss_aggregated:.4f}, MAE: {aggregated_mae:.4f}")
        
        if wandb.run:
            wandb.log({"round": server_round, "global_val_loss": loss_aggregated, "global_val_mae": aggregated_mae, "eval_results_received": len(results)}, commit=True)
        
        return loss_aggregated, metrics_dict

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        logger.info(f"Round {server_round} (FedBN): Server-side 'evaluate' method called by Flower. "
                    "This strategy relies on federated evaluation via 'aggregate_evaluate'. "
                    "No server-side dataset evaluation performed here.")
        
        if wandb.run:
            wandb.log({"round": server_round, "server_evaluate_noop": 1}, commit=True)
        return None


def main():
    logger.info(f"Flower server starting with Python: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Flower (flwr) version: {fl.__version__}")
    logger.info(f"Target device: {DEVICE}")

    initial_model_instance = ScaleDense(nb_filter=8, nb_block=5, use_gender=USE_GENDER_CONFIG).to(DEVICE)
    
    initial_model_path = "./saved_models/initial_model.pth"
    os.makedirs("./saved_models/", exist_ok=True)

    if os.path.exists(initial_model_path):
        logger.info(f"Loading initial model state_dict from {initial_model_path}")
        try:
            # Ensure the model is loaded to the correct device, especially if saved from GPU
            loaded_state_dict = torch.load(initial_model_path, map_location=DEVICE)
            initial_model_instance.load_state_dict(loaded_state_dict, strict=True)
            logger.info("Successfully loaded initial model state_dict.")
        except Exception as e:
            logger.error(f"Failed to load initial_model.pth: {e}. Using fresh weights. Ensure it matches USE_GENDER={USE_GENDER_CONFIG}")
            torch.save(initial_model_instance.state_dict(), initial_model_path)
            logger.info(f"Saved new initial model (due to load failure) to {initial_model_path}")
    else:
        logger.info(f"Initial model {initial_model_path} not found. Using fresh model weights and saving.")
        torch.save(initial_model_instance.state_dict(), initial_model_path)
        logger.info(f"Saved NEW initial model to {initial_model_path}")

    initial_params_ndarrays = get_parameters_from_model(initial_model_instance)
    initial_flwr_params = ndarrays_to_parameters(initial_params_ndarrays)

    strategy = None
    min_clients = FEDCROSS_CONFIG.get("min_clients_per_round", 2) # Use from shared config

    logger.info(f"Selected Federated Learning Strategy: {FL_STRATEGY_CHOICE.upper()}")
    logger.info(f"Minimum clients for fit/eval (K): {min_clients}")
    logger.info(f"Model configured to use gender: {USE_GENDER_CONFIG}")
    logger.info(f"Epochs per client round: {EPOCHS_PER_ROUND}")


    # --- Strategy Initialization ---
    if FL_STRATEGY_CHOICE == "fedavg":
        logger.info("Initializing FedAvg strategy.")
        # FedAvg does not have its own WandB init by default.
        # If you want WandB for FedAvg, you'd typically use a Flower Callback or a global init.
        strategy = FedAvg(
            initial_parameters=initial_flwr_params,
            min_fit_clients=min_clients,
            min_available_clients=min_clients, # Min clients that need to be connected to the server
            min_evaluate_clients=min_clients,
            # fraction_fit=1.0, # Example: fit on all selected clients
            # fraction_evaluate=1.0, # Example: evaluate on all selected clients
            # You can add evaluate_fn and other FedAvg parameters here if needed
        )
        if WANDB_PROJECT_NAME: # Optional: General WandB init if not FedCross
            wandb.init(
                project=WANDB_PROJECT_NAME,
                group=WANDB_RUN_GROUP_NAME,
                name=os.environ.get("WANDB_RUN_NAME_SERVER", f"fedavg_server_run_{np.random.randint(1000,9999)}"),
                config={
                    "strategy": "FedAvg",
                    "min_clients_per_round": min_clients,
                    "epochs_per_round": EPOCHS_PER_ROUND,
                    "use_gender": USE_GENDER_CONFIG,
                    "model_architecture": str(initial_model_instance)
                },
                reinit=True # Allow reinit if FedCrossStrategy also tried to init
            )
            logger.info(f"WandB initialized for FedAvg strategy on project {WANDB_PROJECT_NAME}")

    elif FL_STRATEGY_CHOICE == "fedprox":
        logger.info(f"Initializing FedProx strategy with proximal_mu: {FL_PROXIMAL_MU}.")
        strategy = FedProx(
            initial_parameters=initial_flwr_params,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            min_evaluate_clients=min_clients,
            proximal_mu=FL_PROXIMAL_MU,
            # fraction_fit=1.0,
            # fraction_evaluate=1.0,
        )
        if WANDB_PROJECT_NAME: # Optional: General WandB init if not FedCross
            wandb.init(
                project=WANDB_PROJECT_NAME,
                group=WANDB_RUN_GROUP_NAME,
                name=os.environ.get("WANDB_RUN_NAME_SERVER", f"fedprox_server_run_{np.random.randint(1000,9999)}"),
                config={
                    "strategy": "FedProx",
                    "proximal_mu": FL_PROXIMAL_MU,
                    "min_clients_per_round": min_clients,
                    "epochs_per_round": EPOCHS_PER_ROUND,
                    "use_gender": USE_GENDER_CONFIG,
                    "model_architecture": str(initial_model_instance)

                },
                reinit=True
            )
            logger.info(f"WandB initialized for FedProx strategy on project {WANDB_PROJECT_NAME}")


    elif FL_STRATEGY_CHOICE == "fedcross":
        logger.info("Initializing FedCrossStrategy.")
        logger.info(f"FedCross specific config: alpha={FEDCROSS_CONFIG['alpha']}, collab_strategy={FEDCROSS_CONFIG['collab_strategy']}, dynamic_alpha={FEDCROSS_CONFIG['dynamic_alpha']}")
        strategy = FedCrossStrategy(
            initial_parameters=initial_flwr_params,
            model_template=initial_model_instance, # Crucial for FedCross
            min_fit_clients=min_clients, # K for FedCross
            min_evaluate_clients=min_clients, # K_eval for FedCross, can be different
            min_available_clients=min_clients, # Overall min available
            wandb_project_name=WANDB_PROJECT_NAME, # FedCrossStrategy handles its WandB init
            wandb_run_group_name=WANDB_RUN_GROUP_NAME
        )
        # WandB init is handled inside FedCrossStrategy for FedCross

    else:
        logger.error(f"Unsupported strategy: {FL_STRATEGY_CHOICE}. Choose 'fedavg', 'fedprox', or 'fedcross'. Exiting.")
        return

    if strategy is None: # Should be caught by the else above, but as a safeguard
        logger.error("Strategy was not initialized. Exiting.")
        return

    num_rounds = int(os.environ.get("FL_TOTAL_ROUNDS", 10))
    server_address = os.environ.get("SERVER_ADDRESS", "0.0.0.0:8089")

    logger.info(f"Starting Flower server on {server_address} for {num_rounds} rounds using {FL_STRATEGY_CHOICE.upper()} strategy.")

    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"CRITICAL ERROR during fl.server.start_server: {e}", exc_info=True)
        if wandb.run:
            wandb.log({"server_start_critical_error": str(e)})
        # Re-raise the exception to ensure the script exits with an error status
        raise
    finally: # Ensure WandB run is finished even if errors occur during server run
        if wandb.run:
            wandb.finish()
            logger.info("WandB server run finished.")
        else:
            logger.info("No active WandB run to finish (or WandB was not initialized for this strategy).")


if __name__ == "__main__":
    main()