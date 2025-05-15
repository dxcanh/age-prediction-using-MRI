# central_server/server.py
from flask import Flask, jsonify, request
# Adjust imports: work with state dicts
from model import (load_pretrained_model, save_model, fedcross_aggregation,
                  generate_global_model, initialize_middleware_models, save_middleware_models)
import numpy as np
from threading import Lock
import logging
import time
import random
import copy
import torch # Add torch import

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FL-Server')

app = Flask(__name__)

# --- FedCross Configuration --- (Keep as is)
FEDCROSS_CONFIG = {
    "total_rounds": 100,
    "clients_per_round": 3,  # K value
    "current_round": 0,
    "early_stopping_patience": 10,
    "rounds_without_improvement": 0,
    "best_val_loss": float('inf'),
    "alpha": 0.99,
    "collab_strategy": "lowest_similarity",
    "dynamic_alpha": True,
}

# --- Global State ---
lock = Lock()
client_updates = [] # Store updates received from clients
assigned_models = {}  # Track which middleware model index is assigned to which client_id

# Initialize middleware models (now returns list of state dicts)
initial_model_package = load_pretrained_model() # Load or initialize base model state dict
middleware_models_state_dicts = initialize_middleware_models(
    FEDCROSS_CONFIG["clients_per_round"],
    initial_model_package['model_state_dict']
)
logger.info(f"Initialized {len(middleware_models_state_dicts)} middleware model state dictionaries.")
global_model_package = None # Store the latest global model package {'model_state_dict': ..., 'metadata': ...}


@app.route('/model', methods=['GET'])
def get_model():
    """Send a middleware model state dictionary to the client"""
    global middleware_models_state_dicts, assigned_models

    client_id = request.args.get('client_id', 'unknown')

    with lock:
        # Shuffle indices at the start of a round
        if len(assigned_models) == 0:
            logger.info(f"Round {FEDCROSS_CONFIG['current_round']} - Shuffling middleware model indices")
            middleware_indices = list(range(len(middleware_models_state_dicts)))
            random.shuffle(middleware_indices)
            FEDCROSS_CONFIG["middleware_indices"] = middleware_indices

        # Assign or re-send model
        if client_id in assigned_models:
            model_index = assigned_models[client_id]
            middleware_state_dict = middleware_models_state_dicts[model_index]
            logger.info(f"Re-sending assigned middleware model index {model_index} to client {client_id}")
        else:
            if len(assigned_models) < len(middleware_models_state_dicts):
                model_index = FEDCROSS_CONFIG["middleware_indices"][len(assigned_models)]
                assigned_models[client_id] = model_index
                middleware_state_dict = middleware_models_state_dicts[model_index]
                logger.info(f"Assigned middleware model index {model_index} to client {client_id}")
            else:
                logger.info(f"No available middleware models for client {client_id}, must wait")
                return jsonify({
                    "status": "wait",
                    "message": "All middleware models currently assigned. Please try again later.",
                    "retry_after": 30 # Suggest retry delay
                }), 503 # Service Unavailable

    # Convert numpy arrays to lists for JSON serialization before sending
    serializable_state_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in middleware_state_dict.items()}

    return jsonify({
        "model": serializable_state_dict, # Send the serializable state dict
        "config": {
            "round": FEDCROSS_CONFIG["current_round"],
            "total_rounds": FEDCROSS_CONFIG["total_rounds"],
            "middleware_id": model_index # Send the index assigned
        }
    })

@app.route('/update', methods=['POST'])
def receive_update():
    """Receive updated middleware model state dict from a client"""
    global middleware_models_state_dicts, client_updates, assigned_models, global_model_package

    try:
        data = request.json
        client_id = data.get("client_id", "unknown")
        middleware_id = data.get("middleware_id") # ID/index received by client

        # Validation
        if client_id not in assigned_models or assigned_models[client_id] != middleware_id:
            logger.warning(f"Client {client_id} attempted to update model index {middleware_id} but was assigned {assigned_models.get(client_id)}")
            return jsonify({"status": "error", "message": "Update rejected: Model index mismatch."}), 400

        # Process update (convert lists back to numpy arrays internally)
        received_state_dict = {k: np.array(v) for k, v in data["model"].items()}

        update = {
            "model": received_state_dict, # Store numpy arrays internally
            "dataset_size": data["dataset_size"],
            "validation_loss": data["validation_loss"],
            "middleware_id": middleware_id, # Store which original index this update corresponds to
            "client_id": client_id # Store who sent it
        }

        with lock:
            client_updates.append(update)
            logger.info(f"Received update for middleware model index {middleware_id} from client {client_id} ({len(client_updates)}/{FEDCROSS_CONFIG['clients_per_round']} needed)")

            # Check if enough updates for a round
            if len(client_updates) >= FEDCROSS_CONFIG["clients_per_round"]:
                current_round = FEDCROSS_CONFIG["current_round"]
                logger.info(f"Round {current_round + 1}/{FEDCROSS_CONFIG['total_rounds']} - Starting cross-aggregation")

                # Perform FedCross aggregation (operates on state dicts)
                new_middleware_state_dicts = fedcross_aggregation(
                    client_updates, middleware_models_state_dicts, current_round + 1, FEDCROSS_CONFIG
                )
                middleware_models_state_dicts = new_middleware_state_dicts # Update server state
                FEDCROSS_CONFIG["current_round"] += 1
                round_num = FEDCROSS_CONFIG["current_round"] # Use updated round number


                # Evaluation & Saving logic (operates on state dicts)
                avg_val_loss = sum(update["validation_loss"] for update in client_updates) / len(client_updates)
                logger.info(f"Round {round_num} - Average validation loss: {avg_val_loss:.4f}")

                # Check for improvement
                if avg_val_loss < FEDCROSS_CONFIG["best_val_loss"]:
                     FEDCROSS_CONFIG["best_val_loss"] = avg_val_loss
                     FEDCROSS_CONFIG["rounds_without_improvement"] = 0
                     logger.info(f"New best validation loss: {avg_val_loss:.4f}")
                     # Generate and save best global model package
                     best_global_package = generate_global_model(middleware_models_state_dicts)
                     if best_global_package:
                          best_global_package.setdefault('metadata', {})
                          best_global_package['metadata'].update({
                               'round': round_num, 'timestamp': time.time(), 'best_val_loss': avg_val_loss
                          })
                          save_model(best_global_package, 'best_global_model.pkl')
                else:
                     FEDCROSS_CONFIG["rounds_without_improvement"] += 1
                     logger.info(f"No improvement for {FEDCROSS_CONFIG['rounds_without_improvement']} rounds")


                # Generate and save periodic global model package
                if round_num % 5 == 0 or round_num == FEDCROSS_CONFIG["total_rounds"]:
                    global_model_package = generate_global_model(middleware_models_state_dicts)
                    if global_model_package:
                         global_model_package.setdefault('metadata', {})
                         global_model_package['metadata'].update({'round': round_num, 'timestamp': time.time()})
                         save_model(global_model_package) # Save combined state_dict + metadata

                # Save middleware models state dicts periodically
                if round_num % 10 == 0:
                     save_middleware_models(middleware_models_state_dicts, round_num)

                # Reset for next round
                client_updates.clear()
                assigned_models.clear() # Clear assignments for the new round

                # Check early stopping
                early_stopping = FEDCROSS_CONFIG["rounds_without_improvement"] >= FEDCROSS_CONFIG["early_stopping_patience"]
                if early_stopping:
                     logger.info("Early stopping triggered!")

                return jsonify({
                    "status": "Update received and cross-aggregated",
                    "round": round_num,
                    "early_stopping": early_stopping
                })

        # If not enough updates yet
        return jsonify({"status": "Update received, waiting for more clients"})

    except Exception as e:
        logger.error(f"Error processing update: {str(e)}", exc_info=True) # Log traceback
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get the current status of the federated learning process"""
    with lock: # Protect access to shared config state
        status_data = {
            "status": "running",
            "round": FEDCROSS_CONFIG["current_round"],
            "total_rounds": FEDCROSS_CONFIG["total_rounds"],
            "clients_per_round": FEDCROSS_CONFIG["clients_per_round"],
            "current_client_updates": len(client_updates),
            "middleware_models_count": len(middleware_models_state_dicts),
            "best_val_loss": float(FEDCROSS_CONFIG["best_val_loss"]),
            "collab_strategy": FEDCROSS_CONFIG["collab_strategy"],
            "alpha": FEDCROSS_CONFIG["alpha"],
            "rounds_without_improvement": FEDCROSS_CONFIG["rounds_without_improvement"]
        }
    return jsonify(status_data)


if __name__ == '__main__':
    logger.info("Starting FedCross Federated Learning Server (PyTorch Adaptation)")
    # Ensure model initialization happens before starting server if needed outside first request
    if not middleware_models_state_dicts:
         logger.error("Middleware models failed to initialize!")
    else:
         app.run(host='0.0.0.0', port=5000)