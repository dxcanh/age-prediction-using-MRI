import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple
import copy
import torch
from model_pytorch import get_model, get_model_state_dict, set_model_state_dict

logger = logging.getLogger('FL-Server')

def cross_aggregate(uploaded_model_state_dict, collaborative_model_state_dict, alpha=0.99):
    """Cross-aggregate two model state dictionaries."""
    aggregated_state_dict = {}
    for key in uploaded_model_state_dict.keys():
         # Assuming state dicts contain numpy arrays/lists from client
         uploaded_tensor = torch.tensor(np.array(uploaded_model_state_dict[key]))
         collaborative_tensor = torch.tensor(np.array(collaborative_model_state_dict[key]))
         aggregated_tensor = alpha * uploaded_tensor + (1 - alpha) * collaborative_tensor
         # Store as list/numpy for consistency if needed downstream, or keep as tensor
         aggregated_state_dict[key] = aggregated_tensor.cpu().numpy() # Store as numpy array
         # Or: aggregated_state_dict[key] = aggregated_tensor.tolist()
    return aggregated_state_dict


def fedcross_aggregation(client_updates, middleware_models_state_dicts, round_num, config):
    """
    Implement FedCross aggregation strategy.
    Args:
        client_updates: List of updates containing model state_dicts (as lists/numpy arrays).
        middleware_models_state_dicts: List of current middleware model state_dicts (as lists/numpy arrays).
        round_num: Current round number
        config: FedCross configuration
    Returns:
        List of new middleware model state_dicts after cross-aggregation.
    """
    uploaded_state_dicts = [update["model"] for update in client_updates] # These are dicts of lists/arrays
    new_middleware_state_dicts = []
    

    # For each uploaded model state dict, select a collaborative one and perform cross-aggregation
    for i, uploaded_state_dict in enumerate(uploaded_state_dicts):
        # Select collaborative model state dict
        collaborative_state_dict = select_collaborative_model( # This function needs adaptation if it directly uses model objects
             i, uploaded_state_dicts, round_num, config.get("collab_strategy", "lowest_similarity")
        ) # select_collaborative_model now works on the state dicts directly if cosine_similarity is adapted

        alpha = config.get("alpha", 0.99)
        # Apply dynamic alpha if enabled (logic remains the same)
        if config.get("dynamic_alpha", False):
             max_rounds = config.get("total_rounds", 100); min_alpha = 0.5; target_alpha = alpha
             current_alpha = min_alpha + (target_alpha - min_alpha) * min(1.0, round_num / (max_rounds * 0.5))
             alpha = current_alpha
             logger.info(f"Using dynamic alpha: {alpha:.4f} for round {round_num}")

        # Perform cross-aggregation on the state dicts
        new_middleware_state_dict = cross_aggregate(uploaded_state_dict, collaborative_state_dict, alpha)
        new_middleware_state_dicts.append(new_middleware_state_dict)

    return new_middleware_state_dicts


def generate_global_model(middleware_models_state_dicts):
    """Generate a global model state dict by averaging middleware state dicts."""
    if not middleware_models_state_dicts:
        return None
    global_state_dict = {}
    K = len(middleware_models_state_dicts)

    # Initialize with zeros based on the first model's structure
    first_state_dict = middleware_models_state_dicts[0]
    for key in first_state_dict.keys():
        # Assuming numpy arrays are stored
        global_state_dict[key] = np.zeros_like(np.array(first_state_dict[key]))

    # Simple averaging
    for state_dict in middleware_models_state_dicts:
        for key in global_state_dict.keys():
             global_state_dict[key] += np.array(state_dict[key]) / K

    # Add metadata separately if needed
    global_model_package = {
         'model_state_dict': global_state_dict,
         'metadata': {'generated_from': 'middleware_models'} # Example metadata
    }
    return global_model_package # Return a dict containing state_dict and metadata

def initialize_middleware_models(K, base_model_state_dict=None):
    """Initialize K middleware model state dictionaries."""
    if base_model_state_dict is None:
         # Load or initialize the base model state dict
         initial_model_package = load_pretrained_model() # This now returns {'model_state_dict': ..., 'metadata': ...}
         base_model_state_dict = initial_model_package['model_state_dict']

    middleware_models_state_dicts = []
    for i in range(K):
         # Create a deep copy of the base state dict
         state_dict_copy = copy.deepcopy(base_model_state_dict)
         middleware_models_state_dicts.append(state_dict_copy)
    return middleware_models_state_dicts # Return list of state dicts

def load_pretrained_model(model_path='pretrained_model.pkl'):
    """Load a pre-trained PyTorch model state dict or initialize a new one."""
    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained model state from {model_path}")
        with open(model_path, 'rb') as f:
             # Assume saved file contains the dictionary {'model_state_dict': ..., 'metadata': ...}
             loaded_data = pickle.load(f)
             if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                  return loaded_data
             else:
                  # Handle legacy format if needed, or raise error
                  logger.warning("Loaded model file has unexpected format. Re-initializing.")
                  # Fall through to initialize new model

    # Initialize a new model instance ON CPU to get state dict
    logger.info("No pre-trained model found or format error. Initializing a new CNN3D model.")
    initial_model = get_model().cpu() # Use the same get_model function
    initial_state_dict = {k: v.cpu().numpy() for k, v in initial_model.state_dict().items()} # Store as numpy arrays

    model_package = {
        'model_state_dict': initial_state_dict,
        'metadata': {
            'initialized_from': 'pytorch_cnn3d_random',
            'architecture': 'CNN3D_v1' # Add some versioning/info
        }
    }
    # Optionally save this initial model right away
    save_model(model_package, model_path)
    return model_package


def save_model(model_package, filename='pretrained_model.pkl'):
    """Save the current model package (state_dict + metadata)."""
    if not isinstance(model_package, dict) or 'model_state_dict' not in model_package:
         logger.error("Attempting to save invalid model package format.")
         return

    logger.info(f"Saving model package to {filename}")
    # Ensure state dict tensors are on CPU and converted to numpy/list if not already
    if 'model_state_dict' in model_package:
         state_dict = model_package['model_state_dict']
         serializable_state_dict = {}
         for k, v in state_dict.items():
             if isinstance(v, torch.Tensor):
                 serializable_state_dict[k] = v.cpu().numpy()
             else: # Assume it's already numpy/list
                 serializable_state_dict[k] = v
         model_package['model_state_dict'] = serializable_state_dict


    with open(filename, 'wb') as f:
        pickle.dump(model_package, f)

    # Save timestamped checkpoints (adapt metadata check as needed)
    metadata = model_package.get('metadata', {})
    round_num = metadata.get('round', None)
    if round_num is not None and round_num % 10 == 0:
         timestamped_filename = f'model_round_{round_num}.pkl'
         logger.info(f"Saving checkpoint at round {round_num} to {timestamped_filename}")
         with open(timestamped_filename, 'wb') as f:
             pickle.dump(model_package, f)


def save_middleware_models(middleware_models_state_dicts, round_num):
    """Save the current list of middleware model state dictionaries."""
    directory = 'middleware_models'
    os.makedirs(directory, exist_ok=True)
    filename = f'{directory}/middleware_models_round_{round_num}.pkl'
    logger.info(f"Saving {len(middleware_models_state_dicts)} middleware models state dicts to {filename}")

    # Ensure tensors are converted before saving
    serializable_list = []
    for state_dict in middleware_models_state_dicts:
        serializable_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                serializable_state_dict[k] = v.cpu().numpy()
            else: # Assume already numpy/list
                serializable_state_dict[k] = v
        serializable_list.append(serializable_state_dict)

    with open(filename, 'wb') as f:
         pickle.dump(serializable_list, f) # Save the list of dicts

# --- Adaptation needed for cosine_similarity and select_collaborative_model ---
def cosine_similarity(model_state_dict1, model_state_dict2):
    """Calculate cosine similarity between two model state dictionaries."""
    flat1 = []
    flat2 = []
    # Ensure consistent key iteration
    keys = sorted(model_state_dict1.keys())

    for key in keys:
         # Convert lists/numpy arrays back to tensors for consistent flattening
         # Assuming values are numpy arrays or lists
         tensor1 = torch.tensor(np.array(model_state_dict1[key])).float()
         tensor2 = torch.tensor(np.array(model_state_dict2[key])).float()

         flat1.extend(tensor1.flatten().cpu().numpy())
         flat2.extend(tensor2.flatten().cpu().numpy())

    flat1 = np.array(flat1)
    flat2 = np.array(flat2)

    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1) # Use numpy norm
    norm2 = np.linalg.norm(flat2)

    similarity = dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
    return similarity

def select_collaborative_model(model_index, uploaded_state_dicts, round_num, selection_strategy='lowest_similarity'):
    """Select a collaborative model state dict based on the specified strategy."""
    current_state_dict = uploaded_state_dicts[model_index]
    other_state_dicts = [m for i, m in enumerate(uploaded_state_dicts) if i != model_index]
    K = len(uploaded_state_dicts)

    if not other_state_dicts: # Handle case with only one client update
        logger.warning("Only one client update received, cannot select collaborative model. Returning self.")
        return current_state_dict

    if selection_strategy == 'in_order':
         # Note: Simple in-order might be less effective if K clients don't always participate
         next_original_index = (model_index + (round_num % (K-1)) + 1) % K
         # Find the corresponding state dict in the current list (indices might not match original middleware IDs)
         # This requires knowing the original middleware IDs associated with the updates,
         # which the current `client_updates` structure provides (`update['middleware_id']`).
         # This selection logic needs refinement based on how updates map to original middleware.
         # For simplicity now, just pick the next one in the *current* list of updates.
         next_list_index = (model_index + 1) % len(uploaded_state_dicts) # Simple wrap-around in current list
         return uploaded_state_dicts[next_list_index] # This is not true 'in_order' collaboration

    elif selection_strategy == 'highest_similarity' or selection_strategy == 'lowest_similarity':
         similarities = [cosine_similarity(current_state_dict, other) for other in other_state_dicts]
         if not similarities: return current_state_dict # Should not happen if other_state_dicts is checked

         if selection_strategy == 'highest_similarity':
             best_idx = np.argmax(similarities)
         else: # lowest_similarity
             best_idx = np.argmin(similarities)
         return other_state_dicts[best_idx]

    else:
         logger.warning(f"Unknown selection strategy: {selection_strategy}. Using lowest_similarity instead.")
         similarities = [cosine_similarity(current_state_dict, other) for other in other_state_dicts]
         if not similarities: return current_state_dict
         least_similar_idx = np.argmin(similarities)
         return other_state_dicts[least_similar_idx]