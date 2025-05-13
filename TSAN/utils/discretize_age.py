import torch
import numpy as np

def discretize_age(age, range=5):
    '''
    Discretizes predicted brain age into bins of specified range.

    Args:
        age (torch.Tensor or np.ndarray): Predicted brain age from first stage network.
        range (int, optional): Discretization delta. Defaults to 5.

    Returns:
        torch.Tensor: Discretized predicted brain age with shape (N, 1).
    '''
    # Convert input to numpy array and flatten
    if isinstance(age, torch.Tensor):
        age = age.cpu().numpy()
    age = np.asarray(age).flatten()  # Ensure 1D array

    # Discretize ages
    dis = []
    for i in age:
        value = i // range
        x = i % range
        if x < range / 2:
            discri_age = value * range
        else:
            discri_age = (value + 1) * range
        dis.append(discri_age)
    
    # Convert to numpy array, ensure float32, and add dimension
    dis_age = np.asarray(dis, dtype='float32')
    dis_age = np.expand_dims(dis_age, axis=1)  # Shape (N, 1)
    
    # Convert to torch tensor
    dis_age = torch.from_numpy(dis_age)
    
    return dis_age