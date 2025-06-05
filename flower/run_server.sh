#!/bin/bash

# --- Server Configuration ---
export FL_TOTAL_ROUNDS="10"  
export FL_MIN_CLIENTS="4"  
export SERVER_ADDRESS="0.0.0.0:8089"

# --- Model Configuration (shared by server for initial model and clients) ---
export USE_GENDER="True" 

# --- FedCross Logic Configuration (passed to fedcross_aggregation) ---
export FL_ALPHA="0.9"
export FL_COLLAB_STRATEGY="in_order" # Or "lowest_similarity", "highest_similarity"
export FL_DYNAMIC_ALPHA="True"

# --- WandB Configuration for Server ---
export WANDB_PROJECT_SERVER="fl_tsan_project_server" # Or your preferred WandB project name for server logs
export WANDB_RUN_NAME_SERVER="fedcross_server_run_$(date +%s)"

# --- Paths ---
INITIAL_MODEL_PATH="/home/canhdx/workspace/age-prediction-using-MRI/flower/saved_models/initial_model.pth"

# Check if initial model exists
if [ ! -f "$INITIAL_MODEL_PATH" ]; then
    echo "ERROR: Initial model not found at $INITIAL_MODEL_PATH"
    echo "Please run create_initial_model.py first (after setting USE_GENDER env var if needed for it)."
    exit 1
fi

echo "--- Starting Flower Server ---"
echo "Total Rounds: $FL_TOTAL_ROUNDS"
echo "Min Clients per Round (K for FedCross): $FL_MIN_CLIENTS"
echo "Server Address: $SERVER_ADDRESS"
echo "Using Gender: $USE_GENDER"
echo "FedCross Alpha: $FL_ALPHA"
echo "WandB Project (Server): $WANDB_PROJECT_SERVER"
echo "-----------------------------"


python server.py 