import torch
import os
from tsan_model.ScaleDense import ScaleDense 

USE_GENDER_FOR_INIT = os.environ.get("USE_GENDER", "True").lower() == "true"
NB_FILTER_INIT = 8
NB_BLOCK_INIT = 5
DEVICE_INIT = torch.device("cpu") 

model = ScaleDense(nb_filter=NB_FILTER_INIT, nb_block=NB_BLOCK_INIT, use_gender=USE_GENDER_FOR_INIT)
model.to(DEVICE_INIT) 

# Define the save path
save_dir = "/home/canhdx/workspace/age-prediction-using-MRI/flower/saved_models"
os.makedirs(save_dir, exist_ok=True)
initial_model_save_path = os.path.join(save_dir, "initial_model.pth")

torch.save(model.state_dict(), initial_model_save_path)
print(f"Initial ScaleDense model state_dict saved to: {initial_model_save_path}")
print("Model state_dict keys:", model.state_dict().keys()) 