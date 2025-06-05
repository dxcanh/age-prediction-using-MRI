# fl_tsan_flower_wandb/utils.py
from collections import OrderedDict
from typing import List, Tuple, Dict, Any
import flwr as fl
import torch
import numpy as np
import io

def get_parameters_from_model(model: torch.nn.Module) -> List[np.ndarray]:
    """Return model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters_in_model(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def state_dict_to_bytes(state_dict: OrderedDict) -> bytes:
    """Serialize PyTorch state_dict to bytes."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()

def bytes_to_state_dict(data: bytes) -> OrderedDict:
    """Deserialize PyTorch state_dict from bytes."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location=torch.device("cpu"))

def ndarrays_to_state_dict(ndarrays: List[np.ndarray], template_model: torch.nn.Module) -> OrderedDict:
    """Convert a list of NumPy ndarrays to a PyTorch state_dict, using a template model for keys."""
    params_dict = zip(template_model.state_dict().keys(), ndarrays)
    return OrderedDict({k: torch.tensor(v) for k, v in params_dict})

def state_dict_to_ndarrays(state_dict: OrderedDict) -> List[np.ndarray]:
    """Convert a PyTorch state_dict to a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in state_dict.items()]