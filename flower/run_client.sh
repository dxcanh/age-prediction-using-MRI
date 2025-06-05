#!/bin/bash

# This script is intended to be run for EACH client.
# You will call it multiple times with different CLIENT_ID_NUM.

# --- Client-Specific Configuration (Passed as arguments) ---
CLIENT_ID_NUM=$1
TOTAL_CLIENTS=$2 
SERVER_ADDRESS_CLIENT=${3:-"127.0.0.1:8089"} 

if [ -z "$CLIENT_ID_NUM" ] || [ -z "$TOTAL_CLIENTS" ]; then
  echo "Usage: ./run_client.sh <client_id_num> <total_clients> [server_address]"
  echo "Example: ./run_client.sh 0 3"
  echo "Example: ./run_client.sh 1 3 192.168.1.100:8080"
  exit 1
fi

# --- Shared Model & Data Configuration (Read from environment - set by run_train.sh) ---
# These are now primarily inherited from run_train.sh for consistency.
# If you need client-specific overrides, you can uncomment and set them here.
# export USE_GENDER="True" # Inherited
# export LEARNING_RATE="0.0001" # Inherited
# export EPOCHS_PER_ROUND="3" # Inherited from run_train.sh (was 5) - choose one source
# export BATCH_SIZE="8" # Inherited from run_train.sh (was 16) - choose one source
# export SUBJECT_ID_COLUMN="subject_id" # Inherited
# export LABEL_COLUMN="subject_age" # Inherited
# export GENDER_COLUMN="subject_sex" # Inherited

# --- Paths ---
# UPDATED DATA_DIR to point to the parent of client_X_numpy folders
DATA_DIR="/home/canhdx/workspace/age-prediction-using-MRI/data_per_client"
EXCEL_PATH="/home/canhdx/workspace/age-prediction-using-MRI/label.xlsx" # This is also set in run_train.sh

# --- WandB Configuration for Client ---
# WANDB_PROJECT_CLIENT is set in run_train.sh as CLIENT_WANDB_PROJECT and passed to client.py
# WANDB_RUN_NAME is set and exported in run_train.sh

echo "--- Starting Flower Client $CLIENT_ID_NUM/$((TOTAL_CLIENTS-1)) ---"
echo "INFO: Connecting to Server: $SERVER_ADDRESS_CLIENT"
echo "INFO: Base Data Directory for NumPy files: $DATA_DIR"
echo "INFO: Excel Path for client.py: $EXCEL_PATH" # Excel path used by client.py
echo "INFO: Using Gender (from env): $USE_GENDER"
echo "INFO: Epochs per round (from env): $EPOCHS_PER_ROUND"
echo "INFO: Batch size (from env): $BATCH_SIZE"
echo "INFO: WandB Project (for client.py): $CLIENT_WANDB_PROJECT" # Value from run_train.sh
echo "INFO: WandB Run Name (from env): $WANDB_RUN_NAME"
echo "--------------------------------"

# Client-side delay, run_train.sh already waits for server
# echo "Waiting for a few seconds (client-side delay)..." 
# sleep 5 

# Virtual environment is activated in run_train.sh

python client.py \
  --client-id-num=$CLIENT_ID_NUM \
  --sorter-checkpoint-path="/home/canhdx/workspace/age-prediction-using-MRI/flower/tsan_model/best_lstmla_slen_8.pth.tar"\
  --total-clients=$TOTAL_CLIENTS \
  --server-address="$SERVER_ADDRESS_CLIENT" \
  --data-dir="$DATA_DIR/client_${CLIENT_ID_NUM}_numpy" \
  --excel-path="$EXCEL_PATH" \
  --wandb-project="$CLIENT_WANDB_PROJECT" # Use var from run_train.sh