import flwr as fl
from flwr.common import (
    FitRes, Metrics, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy 
from flwr.server.strategy import FedAvg, FedProx 

import torch
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import os
import wandb
import logging

from tsan_model.ScaleDense import ScaleDense
from central_server_logic.fedcross_logic import fedcross_aggregation, generate_global_model_pytorch
from central_server_logic.fedbn import fedbn_aggregation, generate_global_model_fedbn
from utils import ndarrays_to_state_dict, state_dict_to_ndarrays

logger = logging.getLogger("FlowerServer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cpu")

USE_GENDER_CONFIG = os.environ.get("USE_GENDER", "True").lower() == "true"
EPOCHS_PER_ROUND = int(os.environ.get("EPOCHS_PER_ROUND", 3))
FL_STRATEGY_CHOICE = os.environ.get("FL_STRATEGY", "fedavg").lower() 
FL_PROXIMAL_MU = float(os.environ.get("FL_PROXIMAL_MU", 0.1)) 

FEDCROSS_CONFIG = {
    "alpha": float(os.environ.get("FL_ALPHA", 0.9)),
    "collab_strategy": os.environ.get("FL_COLLAB_STRATEGY", "in_order"),
    "dynamic_alpha": os.environ.get("FL_DYNAMIC_ALPHA", "True").lower() == "true",
    "min_clients_per_round": int(os.environ.get("FL_MIN_CLIENTS", 2)) 
}

WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_SERVER", "fl_tsan_project_server")
WANDB_RUN_GROUP_NAME = os.environ.get("WANDB_RUN_GROUP_SERVER", "federated_tsan_server_fedhead")

def main():
    logger.info(f"Flower server (FedHead compatible) starting with Python: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Flower (flwr) version: {fl.__version__}")
    logger.info(f"Target device: {DEVICE}")

    temp_full_model_for_structure = ScaleDense(nb_filter=8, nb_block=5, use_gender=USE_GENDER_CONFIG).to(DEVICE)
    
    initial_model_path = os.environ.get("INITIAL_MODEL_PATH", "./saved_models/initial_model.pth")
    os.makedirs(os.path.dirname(initial_model_path), exist_ok=True)

    initial_shared_ndarrays = []
    shared_prefixes = ("pre.", "block.", "gap.", "deep_fc.")

    if os.path.exists(initial_model_path):
        logger.info(f"Loading initial FULL model state_dict from {initial_model_path} to extract SHARED part.")
        try:
            loaded_full_state_dict = torch.load(initial_model_path, map_location=DEVICE)
            temp_full_model_for_structure.load_state_dict(loaded_full_state_dict, strict=True) 
            ordered_shared_keys = [
                key for key in temp_full_model_for_structure.state_dict().keys() 
                if key.startswith(shared_prefixes)
            ]
            for key in ordered_shared_keys:
                initial_shared_ndarrays.append(loaded_full_state_dict[key].cpu().numpy())
            logger.info(f"Successfully loaded full model and extracted {len(initial_shared_ndarrays)} shared parameter tensors.")
        except Exception as e:
            logger.error(f"Failed to load/process initial_model.pth: {e}. Will use fresh SHARED weights from a new temp model.")
            initial_shared_ndarrays = []

    if not initial_shared_ndarrays:
        logger.info(f"Generating fresh SHARED initial parameters as {initial_model_path} was not loaded or processed.")
        if not os.path.exists(initial_model_path):
             torch.save(temp_full_model_for_structure.state_dict(), initial_model_path)
             logger.info(f"Saved NEW full initial model to {initial_model_path}")
        current_full_state_dict = temp_full_model_for_structure.state_dict()
        ordered_shared_keys = [
            key for key in current_full_state_dict.keys() 
            if key.startswith(shared_prefixes)
        ]
        for key in ordered_shared_keys:
            initial_shared_ndarrays.append(current_full_state_dict[key].cpu().numpy())
        logger.info(f"Using {len(initial_shared_ndarrays)} shared parameter tensors from a fresh temp model instance.")

    initial_flwr_shared_params = ndarrays_to_parameters(initial_shared_ndarrays)

    strategy = None
    min_clients_for_strategy = int(os.environ.get("FL_MIN_CLIENTS", 2))

    logger.info(f"Selected Federated Learning Strategy for FedHead: {FL_STRATEGY_CHOICE.upper()}")
    logger.info(f"Minimum clients for fit/eval (K): {min_clients_for_strategy}")
    logger.info(f"Model configured to use gender (by clients): {USE_GENDER_CONFIG}")
    logger.info(f"Epochs per client round (for client config): {EPOCHS_PER_ROUND}")

    wandb_base_config = {
        "min_clients_per_round": min_clients_for_strategy,
        "epochs_per_round_client": EPOCHS_PER_ROUND,
        "use_gender_client": USE_GENDER_CONFIG,
        "num_shared_parameters": len(initial_shared_ndarrays), 
    }

    if FL_STRATEGY_CHOICE == "fedavg":
        logger.info("Initializing FedAvg strategy for FedHead (aggregating shared params).")
        strategy = FedAvg(
            initial_parameters=initial_flwr_shared_params,
            min_fit_clients=min_clients_for_strategy,
            min_available_clients=min_clients_for_strategy, 
            min_evaluate_clients=min_clients_for_strategy,
        )
        if WANDB_PROJECT_NAME:
            wandb.init(
                project=WANDB_PROJECT_NAME,
                group=WANDB_RUN_GROUP_NAME,
                name=os.environ.get("WANDB_RUN_NAME_SERVER", f"fedavg_fedhead_server_run_{np.random.randint(1000,9999)}"),
                config={**wandb_base_config, "strategy": "FedAvg_FedHead"},
                reinit=True
            )
            logger.info(f"WandB initialized for FedAvg (FedHead) strategy on project {WANDB_PROJECT_NAME}")

    elif FL_STRATEGY_CHOICE == "fedprox":
        logger.info(f"Initializing FedProx strategy for FedHead with proximal_mu: {FL_PROXIMAL_MU} (aggregating shared params).")
        strategy = FedProx(
            initial_parameters=initial_flwr_shared_params,
            min_fit_clients=min_clients_for_strategy,
            min_available_clients=min_clients_for_strategy,
            min_evaluate_clients=min_clients_for_strategy,
            proximal_mu=FL_PROXIMAL_MU,
        )
        if WANDB_PROJECT_NAME:
            wandb.init(
                project=WANDB_PROJECT_NAME,
                group=WANDB_RUN_GROUP_NAME,
                name=os.environ.get("WANDB_RUN_NAME_SERVER", f"fedprox_fedhead_server_run_{np.random.randint(1000,9999)}"),
                config={**wandb_base_config, "strategy": "FedProx_FedHead", "proximal_mu": FL_PROXIMAL_MU},
                reinit=True
            )
            logger.info(f"WandB initialized for FedProx (FedHead) strategy on project {WANDB_PROJECT_NAME}")
    
    else:
        logger.error(f"Unsupported strategy: {FL_STRATEGY_CHOICE} for FedHead. Choose 'fedavg' or 'fedprox'. Exiting.")
        return

    if strategy is None: 
        logger.error("Strategy was not initialized. Exiting.")
        return

    num_rounds = int(os.environ.get("FL_TOTAL_ROUNDS", 10))
    server_address = os.environ.get("SERVER_ADDRESS", "0.0.0.0:8089")

    logger.info(f"Starting Flower server on {server_address} for {num_rounds} rounds using {FL_STRATEGY_CHOICE.upper()} (FedHead) strategy.")

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
        raise
    finally: 
        if wandb.run:
            wandb.finish()
            logger.info("WandB server run finished.")
        else:
            logger.info("No active WandB run to finish (or WandB was not initialized).")

if __name__ == "__main__":
    main()
