import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple, Union, Set
import copy
import torch
import torch.nn.functional as F

from .fedcross_logic import fedcross_aggregation

logger = logging.getLogger('FL-Server')
logging.basicConfig(level=logging.INFO)

def identify_bn_layers(state_dict: Dict[str, torch.Tensor]) -> Set[str]:
    """
    Identify batch normalization layer parameters in a model state dict.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Set of parameter names that belong to batch normalization layers
    """
    bn_keys = set()
    
    # Common BatchNorm parameter patterns
    bn_patterns = [
        'bn', 'batch_norm', 'batchnorm', 'norm',
        '.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked'
    ]
    
    for key in state_dict.keys():
        key_lower = key.lower()
        
        # Check if this is a BatchNorm parameter
        is_bn = False
        
        # Pattern 1: Contains 'bn' or batch norm indicators
        if any(pattern in key_lower for pattern in ['bn', 'batch_norm', 'batchnorm']):
            is_bn = True
        
        # Pattern 2: Layer normalization or group normalization
        elif any(pattern in key_lower for pattern in ['layer_norm', 'group_norm', 'instance_norm']):
            is_bn = True
            
        # Pattern 3: Common BatchNorm parameter suffixes with layer indicators
        elif any(key_lower.endswith(suffix) for suffix in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']):
            # Check if the layer name suggests it's a normalization layer
            layer_name = key.rsplit('.', 1)[0] if '.' in key else key
            if any(norm_indicator in layer_name.lower() for norm_indicator in ['bn', 'norm', 'batch']):
                is_bn = True
        
        if is_bn:
            bn_keys.add(key)
    
    logger.info(f"Identified {len(bn_keys)} BatchNorm parameters: {sorted(list(bn_keys))}")
    return bn_keys

def separate_bn_and_other_params(
    state_dict: Dict[str, torch.Tensor],
    bn_keys: Set[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Separate BatchNorm parameters from other parameters.
    
    Args:
        state_dict: Complete model state dictionary
        bn_keys: Set of keys corresponding to BatchNorm parameters
        
    Returns:
        Tuple of (bn_params, other_params)
    """
    bn_params = {}
    other_params = {}
    
    for key, value in state_dict.items():
        if key in bn_keys:
            bn_params[key] = value
        else:
            other_params[key] = value
    
    return bn_params, other_params

def fedbn_aggregate_non_bn_params(
    client_updates: List[Dict[str, Any]],
    bn_keys: Set[str],
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Aggregate only non-BatchNorm parameters using FedAvg.
    
    Args:
        client_updates: List of client model updates
        bn_keys: Set of BatchNorm parameter keys to exclude
        device: Device to perform computation on
        
    Returns:
        Aggregated non-BatchNorm parameters
    """
    if not client_updates:
        logger.error("Cannot aggregate from empty client updates.")
        return {}
    
    # Extract non-BN parameters from all clients
    non_bn_params_list = []
    for update in client_updates:
        _, non_bn_params = separate_bn_and_other_params(update['model'], bn_keys)
        non_bn_params_list.append(non_bn_params)
    
    # Perform FedAvg on non-BN parameters
    aggregated_params = {}
    first_params = non_bn_params_list[0]
    
    with torch.no_grad():
        # Initialize aggregated parameters
        for key, param in first_params.items():
            aggregated_params[key] = torch.zeros_like(param, device=device)
        
        # Sum all client parameters
        for non_bn_params in non_bn_params_list:
            for key in aggregated_params.keys():
                if key in non_bn_params:
                    param_tensor = non_bn_params[key].to(device)
                    if aggregated_params[key].dtype != param_tensor.dtype:
                        aggregated_params[key] = aggregated_params[key].float()
                        param_tensor = param_tensor.float()
                    aggregated_params[key] += param_tensor
                else:
                    logger.warning(f"Key {key} missing in client non-BN parameters.")
        
        # Average the parameters
        num_clients = len(non_bn_params_list)
        for key in aggregated_params.keys():
            if aggregated_params[key].is_floating_point():
                aggregated_params[key] /= num_clients
    
    logger.info(f"Aggregated {len(aggregated_params)} non-BatchNorm parameters from {num_clients} clients.")
    return aggregated_params

def fedbn_create_personalized_models(
    client_updates: List[Dict[str, Any]],
    aggregated_non_bn_params: Dict[str, torch.Tensor],
    bn_keys: Set[str],
    device: torch.device = torch.device("cpu")
) -> List[Dict[str, torch.Tensor]]:
    """
    Create personalized models by combining aggregated non-BN params with local BN params.
    
    Args:
        client_updates: List of client model updates
        aggregated_non_bn_params: Globally aggregated non-BatchNorm parameters
        bn_keys: Set of BatchNorm parameter keys
        device: Device to perform computation on
        
    Returns:
        List of personalized model state dictionaries
    """
    personalized_models = []
    
    for update in client_updates:
        client_bn_params, _ = separate_bn_and_other_params(update['model'], bn_keys)
        
        # Combine global non-BN params with local BN params
        personalized_model = {}
        
        # Add aggregated non-BN parameters
        for key, param in aggregated_non_bn_params.items():
            personalized_model[key] = param.clone().detach().to(device)
        
        # Add local BN parameters
        for key, param in client_bn_params.items():
            personalized_model[key] = param.clone().detach().to(device)
        
        personalized_models.append(personalized_model)
        logger.debug(f"Created personalized model for client {update['middleware_id']} with "
                    f"{len(aggregated_non_bn_params)} global params and {len(client_bn_params)} local BN params.")
    
    return personalized_models

def fedbn_aggregation(
    client_updates: List[Dict[str, Any]],
    K: int,
    round_num: int,
    config: Dict[str, Any],
    device: torch.device = torch.device("cpu")
) -> List[Dict[str, torch.Tensor]]:
    """
    FedBN aggregation: Keep BatchNorm parameters local, aggregate others globally.
    
    Args:
        client_updates: List of client model updates with 'middleware_id' and 'model'
        K: Number of clients
        round_num: Current round number
        config: Configuration dictionary
        device: Device to perform computation on
        
    Returns:
        List of personalized model state dictionaries
    """
    if len(client_updates) != K:
        logger.warning(f"Expected K={K} client updates, but received {len(client_updates)}.")
    
    if not client_updates:
        logger.error("No client updates provided for FedBN aggregation.")
        return []
    
    # Identify BatchNorm parameters from the first client's model
    first_model = client_updates[0]['model']
    bn_keys = identify_bn_layers(first_model)
    
    if not bn_keys:
        logger.warning("No BatchNorm parameters detected. FedBN will behave like standard FedAvg.")
    
    # Aggregate non-BatchNorm parameters
    aggregated_non_bn_params = fedbn_aggregate_non_bn_params(
        client_updates, bn_keys, device
    )
    
    # Create personalized models
    personalized_models = fedbn_create_personalized_models(
        client_updates, aggregated_non_bn_params, bn_keys, device
    )
    
    logger.info(f"FedBN aggregation completed for round {round_num}. "
               f"Generated {len(personalized_models)} personalized models.")
    
    return personalized_models

def fedbn_with_cross_aggregation(
    client_updates: List[Dict[str, Any]],
    K: int,
    round_num: int,
    config: Dict[str, Any],
    device: torch.device = torch.device("cpu")
) -> List[Dict[str, torch.Tensor]]:
    """
    Hybrid approach: Apply FedBN principles with cross-aggregation techniques.
    
    This function first applies cross-aggregation to non-BN parameters,
    then combines with local BN parameters.
    """
    if len(client_updates) != K:
        logger.warning(f"Expected K={K} client updates, but received {len(client_updates)}.")
    
    if not client_updates:
        logger.error("No client updates provided for FedBN with cross-aggregation.")
        return []
    
    # Identify BatchNorm parameters
    first_model = client_updates[0]['model']
    bn_keys = identify_bn_layers(first_model)
    
    # Create updates with only non-BN parameters for cross-aggregation
    non_bn_updates = []
    for update in client_updates:
        _, non_bn_params = separate_bn_and_other_params(update['model'], bn_keys)
        non_bn_updates.append({
            'middleware_id': update['middleware_id'],
            'model': non_bn_params
        })
    
    try:
        cross_aggregated_non_bn = fedcross_aggregation(
            non_bn_updates, K, round_num, config, device
        )
    except:
        # Fallback to simple aggregation if cross-aggregation fails
        logger.warning("Cross-aggregation failed, falling back to simple FedAvg for non-BN params.")
        aggregated_non_bn_params = fedbn_aggregate_non_bn_params(
            client_updates, bn_keys, device
        )
        cross_aggregated_non_bn = [aggregated_non_bn_params] * K
    
    # Combine cross-aggregated non-BN params with local BN params
    personalized_models = []
    for i, update in enumerate(client_updates):
        client_bn_params, _ = separate_bn_and_other_params(update['model'], bn_keys)
        
        # Combine cross-aggregated non-BN params with local BN params
        personalized_model = {}
        
        # Add cross-aggregated non-BN parameters
        if i < len(cross_aggregated_non_bn):
            for key, param in cross_aggregated_non_bn[i].items():
                personalized_model[key] = param.clone().detach().to(device)
        
        # Add local BN parameters
        for key, param in client_bn_params.items():
            personalized_model[key] = param.clone().detach().to(device)
        
        personalized_models.append(personalized_model)
    
    logger.info(f"FedBN with cross-aggregation completed for round {round_num}. "
               f"Generated {len(personalized_models)} personalized models.")
    
    return personalized_models

def generate_global_model_fedbn(
    middleware_models_state_dicts: List[Dict[str, torch.Tensor]],
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Generate a global model for FedBN by averaging non-BN parameters
    and using BN parameters from the first model.
    """
    if not middleware_models_state_dicts:
        logger.error("Cannot generate global model from empty list of middleware models.")
        return None
    
    first_model = middleware_models_state_dicts[0]
    bn_keys = identify_bn_layers(first_model)
    
    # Separate BN and non-BN parameters
    global_state_dict = {}
    
    # For non-BN parameters, compute average
    non_bn_params_list = []
    for state_dict in middleware_models_state_dicts:
        _, non_bn_params = separate_bn_and_other_params(state_dict, bn_keys)
        non_bn_params_list.append(non_bn_params)
    
    # Average non-BN parameters
    if non_bn_params_list:
        first_non_bn = non_bn_params_list[0]
        with torch.no_grad():
            for key, param in first_non_bn.items():
                global_state_dict[key] = torch.zeros_like(param, device=device)
            
            for non_bn_params in non_bn_params_list:
                for key in global_state_dict.keys():
                    if key in non_bn_params:
                        param_tensor = non_bn_params[key].to(device)
                        if global_state_dict[key].dtype != param_tensor.dtype:
                            global_state_dict[key] = global_state_dict[key].float()
                            param_tensor = param_tensor.float()
                        global_state_dict[key] += param_tensor
            
            # Average
            num_models = len(non_bn_params_list)
            for key in global_state_dict.keys():
                if global_state_dict[key].is_floating_point():
                    global_state_dict[key] /= num_models
    
    # For BN parameters, use the first model's BN parameters
    first_bn_params, _ = separate_bn_and_other_params(first_model, bn_keys)
    for key, param in first_bn_params.items():
        global_state_dict[key] = param.clone().detach().to(device)
    
    logger.info(f"Generated global model with {len(global_state_dict)} parameters "
               f"({len(first_bn_params)} BN params from first model, "
               f"{len(global_state_dict) - len(first_bn_params)} averaged non-BN params).")
    
    return global_state_dict

def save_fedbn_models(
    middleware_models_state_dicts: List[Dict[str, torch.Tensor]], 
    round_num: int,
    save_bn_separately: bool = True
):
    """
    Save FedBN models, optionally separating BN and non-BN parameters.
    """
    directory = 'fedbn_models_torch'
    os.makedirs(directory, exist_ok=True)
    
    if save_bn_separately and middleware_models_state_dicts:
        # Save BN and non-BN parameters separately
        first_model = middleware_models_state_dicts[0]
        bn_keys = identify_bn_layers(first_model)
        
        bn_params_list = []
        non_bn_params_list = []
        
        for state_dict in middleware_models_state_dicts:
            bn_params, non_bn_params = separate_bn_and_other_params(state_dict, bn_keys)
            bn_params_list.append({k: v.cpu() for k, v in bn_params.items()})
            non_bn_params_list.append({k: v.cpu() for k, v in non_bn_params.items()})
        
        # Save BN parameters (personalized)
        bn_filename = f'{directory}/bn_params_round_{round_num}.pth'
        torch.save(bn_params_list, bn_filename)
        logger.info(f"Saved {len(bn_params_list)} client BN parameters to {bn_filename}")
        
        # Save non-BN parameters (shared)
        non_bn_filename = f'{directory}/non_bn_params_round_{round_num}.pth'
        torch.save(non_bn_params_list, non_bn_filename)
        logger.info(f"Saved {len(non_bn_params_list)} client non-BN parameters to {non_bn_filename}")
    
    # Also save complete models
    filename = f'{directory}/complete_models_round_{round_num}.pth'
    serializable_list = [{k: v.cpu() for k, v in sd.items()} for sd in middleware_models_state_dicts]
    
    try:
        torch.save(serializable_list, filename)
        logger.info(f"Saved {len(serializable_list)} complete FedBN models to {filename}")
    except Exception as e:
        logger.error(f"Error saving FedBN models: {e}")