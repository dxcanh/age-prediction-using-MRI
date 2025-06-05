import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple, Union
import copy
import torch
import torch.nn.functional as F

logger = logging.getLogger('FL-Server')
logging.basicConfig(level=logging.INFO)

def cross_aggregate(
    uploaded_model_state_dict: Dict[str, torch.Tensor],
    collaborative_model_state_dict: Dict[str, torch.Tensor],
    alpha: float,
    device: torch.device = torch.device('cpu')
)  -> Dict[str, torch.Tensor]:
    if not (0.5 <= alpha <= 1.0):
        logger.warning(f"Alpha {alpha} is outside the range [0.5, 1.0]")

    aggregated_state_dict = {}
    with torch.no_grad(): #ensure gradients are not tracked
        for key in uploaded_model_state_dict.keys():
            uploaded_tensor = uploaded_model_state_dict[key].to(device)
            collaborative_tensor = collaborative_model_state_dict[key].to(device)

            if uploaded_tensor.dtype != collaborative_tensor.dtype:
                if uploaded_tensor.is_floating_point() or collaborative_tensor.is_floating_point():
                     uploaded_tensor = uploaded_tensor.float()
                     collaborative_tensor = collaborative_tensor.float()
                else:
                    logger.warning(f"Aggregating non-floating point tensors for key {key}")
                    aggregated_state_dict[key] = uploaded_tensor.clone()
                    continue

            #weighted aggregation
            aggregated_tensor = alpha * uploaded_tensor + (1 - alpha) * collaborative_tensor
            aggregated_state_dict[key] = aggregated_tensor
    
    return aggregated_state_dict


def cosine_similarity_pytorch(
    model_state_dict1: Dict[str, torch.Tensor],
    model_state_dict2: Dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu")
) -> float:
    vec1 = []
    vec2 = []
    with torch.no_grad():
        keys = sorted([k for k, v in model_state_dict1.items() if v.is_floating_point()])
        for key in keys:
            if key in model_state_dict2 and model_state_dict2[key].is_floating_point():
                vec1.append(model_state_dict1[key].to(device).flatten())
                vec2.append(model_state_dict2[key].to(device).flatten())
            else:
                logger.debug(f"Skipping key {key} for similarity calculation (missing or non-float).")

        if not vec1:
             return 0.0

        flat_vec1 = torch.cat(vec1)
        flat_vec2 = torch.cat(vec2)
        similarity = F.cosine_similarity(flat_vec1.unsqueeze(0), flat_vec2.unsqueeze(0), dim=1, eps=1e-8)

    return similarity.item()

def select_collaborative_model(
    target_middleware_index: int,
    uploaded_updates: List[Dict[str, Any]],
    round_num: int,
    K: int,
    selection_strategy: str = 'lowest_similarity',
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    if K < 2:
        raise ValueError("Cannot select collaborative model when K < 2.")

    target_state_dict = None
    for update in uploaded_updates:
        if update['middleware_id'] == target_middleware_index:
            target_state_dict = update['model']
            break
    if target_state_dict is None:
         raise ValueError(f"Target middleware index {target_middleware_index} not found in uploaded updates.")

    other_updates = [upd for upd in uploaded_updates if upd['middleware_id'] != target_middleware_index]

    if not other_updates:
         logger.warning(f"Only one model update found (ID: {target_middleware_index}). Returning self as collaborator.")
         return target_state_dict

    if selection_strategy == 'in_order':
        collaborative_original_index = (target_middleware_index + (round_num % (K - 1)) + 1) % K

        for update in uploaded_updates:
            if update['middleware_id'] == collaborative_original_index:
                if collaborative_original_index == target_middleware_index:
                    logger.warning(f"In-order selected self (Index {target_middleware_index}) for K={K}, round {round_num}. Check logic.")
                    if other_updates: return other_updates[0]['model']
                    else: return target_state_dict

                logger.debug(f"In-order: Target {target_middleware_index}, Collab Original Index {collaborative_original_index}")
                return update['model']
        logger.error(f"In-order collaborative model index {collaborative_original_index} not found in updates. Falling back.")
        return other_updates[0]['model']

    elif selection_strategy in ['highest_similarity', 'lowest_similarity']:
        similarities = []
        valid_other_updates = []
        for other_update in other_updates:
            try:
                sim = cosine_similarity_pytorch(target_state_dict, other_update['model'], device=device)
                similarities.append(sim)
                valid_other_updates.append(other_update)
            except Exception as e:
                 logger.error(f"Error calculating similarity between {target_middleware_index} and {other_update['middleware_id']}: {e}")

        if not similarities:
             logger.warning(f"Could not calculate similarities for target {target_middleware_index}. Returning first other.")
             return other_updates[0]['model']

        if selection_strategy == 'highest_similarity':
            best_idx = np.argmax(similarities)
        else:
            best_idx = np.argmin(similarities)

        selected_update = valid_other_updates[best_idx]
        logger.debug(f"{selection_strategy}: Target {target_middleware_index}, Selected Collab ID {selected_update['middleware_id']}, Sim: {similarities[best_idx]:.4f}")
        return selected_update['model']

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

def fedcross_aggregation(
    client_updates: List[Dict[str, Any]],
    K: int,
    round_num: int,
    config: Dict[str, Any],
    device: torch.device = torch.device("cpu")
) -> List[Dict[str, torch.Tensor]]:
    if len(client_updates) != K:
         logger.warning(f"Expected K={K} client updates, but received {len(client_updates)}. Aggregation might be incomplete or fail.")

    uploaded_state_dicts_map = {update['middleware_id']: update['model'] for update in client_updates}

    new_middleware_state_dicts = [None] * K

    base_alpha = config.get("alpha", 0.9)
    collab_strategy = config.get("collab_strategy", "lowest_similarity")

    alpha = base_alpha
    if config.get("dynamic_alpha", False):
        max_rounds = config.get("total_rounds", 100)
        min_alpha = 0.5
        target_alpha = base_alpha
        current_progress = min(1.0, (round_num + 1) / max_rounds)
        alpha = min_alpha + (target_alpha - min_alpha) * current_progress
        logger.info(f"Using dynamic alpha: {alpha:.4f} for round {round_num+1}/{max_rounds}")

    for i in range(K):
        if i not in uploaded_state_dicts_map:
             logger.error(f"Middleware model {i} update missing. Cannot perform cross-aggregation for it.")
             continue

        current_state_dict = uploaded_state_dicts_map[i]

        try:
            collaborative_state_dict = select_collaborative_model(
                target_middleware_index=i,
                uploaded_updates=client_updates,
                round_num=round_num,
                K=K,
                selection_strategy=collab_strategy,
                device=device
            )

            new_state_dict = cross_aggregate(
                uploaded_model_state_dict=current_state_dict,
                collaborative_model_state_dict=collaborative_state_dict,
                alpha=alpha,
                device=device
            )
            new_middleware_state_dicts[i] = new_state_dict

        except Exception as e:
            logger.error(f"Failed cross-aggregation for middleware index {i}: {e}")
            new_middleware_state_dicts[i] = current_state_dict.copy()

    final_state_dicts = [sd for sd in new_middleware_state_dicts if sd is not None]
    if len(final_state_dicts) != K:
         logger.warning(f"Aggregation resulted in {len(final_state_dicts)} models, expected {K}.")

    return final_state_dicts

def generate_global_model_pytorch(
    middleware_models_state_dicts: List[Dict[str, torch.Tensor]],
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    if not middleware_models_state_dicts:
        logger.error("Cannot generate global model from empty list of middleware models.")
        return None
    K = len(middleware_models_state_dicts)
    global_state_dict = {}
    first_state_dict = middleware_models_state_dicts[0]

    with torch.no_grad():
        for key, param in first_state_dict.items():
            global_state_dict[key] = torch.zeros_like(param, device=device)

        for state_dict in middleware_models_state_dicts:
            for key in global_state_dict.keys():
                 if key in state_dict:
                     if global_state_dict[key].dtype != state_dict[key].dtype:
                           global_state_dict[key] = global_state_dict[key].float()
                           global_state_dict[key] += state_dict[key].to(device).float()
                     else:
                          global_state_dict[key] += state_dict[key].to(device)
                 else:
                     logger.warning(f"Key {key} missing in one of the middleware models during global averaging.")

        for key in global_state_dict.keys():
             if global_state_dict[key].is_floating_point():
                global_state_dict[key] /= K
             else:
                global_state_dict[key] = first_state_dict[key].to(device)

    return global_state_dict

def state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: v.cpu().numpy() for k, v in state_dict.items()}

def state_dict_to_torch(
     state_dict_numpy: Dict[str, np.ndarray],
     device: torch.device = torch.device("cpu")
     ) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in state_dict_numpy.items()}

def initialize_middleware_models(
     K: int,
     base_model_path: str = 'initial_model.pth',
     model_loader_func = None,
     device: torch.device = torch.device("cpu")
     ) -> List[Dict[str, torch.Tensor]]:
    if os.path.exists(base_model_path):
        logger.info(f"Loading base model state dict from {base_model_path}")
        base_model_state_dict = torch.load(base_model_path, map_location=device)
        if not isinstance(base_model_state_dict, dict):
             logger.error(f"Loaded file {base_model_path} is not a state_dict. Re-initializing.")
             base_model_state_dict = None
        else:
             base_model_state_dict = {k: v.to(device) for k, v in base_model_state_dict.items()}

    else:
        logger.info(f"Base model {base_model_path} not found. Initializing a new model.")
        if model_loader_func is None:
            raise ValueError("model_loader_func is required to initialize a new model.")
        initial_model = model_loader_func().to(device)
        base_model_state_dict = initial_model.state_dict()
        torch.save(base_model_state_dict, base_model_path)
        logger.info(f"Saved newly initialized model state dict to {base_model_path}")

    middleware_models_state_dicts = []
    for _ in range(K):
        state_dict_copy = {k: v.clone().detach() for k, v in base_model_state_dict.items()}
        middleware_models_state_dicts.append(state_dict_copy)

    logger.info(f"Initialized {K} middleware models.")
    return middleware_models_state_dicts

def save_middleware_models(middleware_models_state_dicts: List[Dict[str, torch.Tensor]], round_num: int):
    directory = 'middleware_models_torch'
    os.makedirs(directory, exist_ok=True)
    filename = f'{directory}/middleware_models_round_{round_num}.pth'
    logger.info(f"Saving {len(middleware_models_state_dicts)} middleware models state dicts to {filename}")

    serializable_list = [{k: v.cpu() for k, v in sd.items()} for sd in middleware_models_state_dicts]

    try:
        torch.save(serializable_list, filename)
    except Exception as e:
        logger.error(f"Error saving middleware models: {e}")