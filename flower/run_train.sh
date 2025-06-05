#!/bin/bash

echo "--- FL Experiment Setup ---"

# --- CHOOSE YOUR STRATEGY ---
# Options: "fedavg", "fedprox", "fedcross"
export FL_STRATEGY="fedprox" # <<<< ------ SET YOUR DESIRED STRATEGY HERE

# --- GPU Configuration ---
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader || echo 0)
echo "Number of GPUs detected by nvidia-smi: $NUM_AVAILABLE_GPUS"

# --- Shared Configuration (Used by Server & Clients) ---
# These will also be available to run_server.sh if it needs them and doesn't redefine
export USE_GENDER="True"
export EPOCHS_PER_ROUND="5"
export LEARNING_RATE="1e-5"
export BATCH_SIZE="8"
export SUBJECT_ID_COLUMN="subject_id"
export LABEL_COLUMN="subject_age"
export GENDER_COLUMN="subject_sex"

# --- Server Configuration (Potentially overridden or used by run_server.sh) ---
export FL_TOTAL_ROUNDS="10" # run_server.sh has 10
export FL_MIN_CLIENTS="4"   # run_server.sh has 2
export SERVER_ADDRESS="0.0.0.0:8089" # Consistent

# --- Algorithm-Specific Server Parameters (Potentially overridden or used by run_server.sh) ---
export FL_PROXIMAL_MU="0.1" # For FedProx
export FL_ALPHA="0.9" # For FedCross (run_server.sh also sets this)
export FL_COLLAB_STRATEGY="in_order" # For FedCross (run_server.sh also sets this)
export FL_DYNAMIC_ALPHA="True" # For FedCross (run_server.sh also sets this)

# --- WandB Configuration (Server - Potentially overridden by run_server.sh) ---
export CLIENT_WANDB_PROJECT="fl_tsan_project_clients_exp"
export WANDB_PROJECT_SERVER="fl_tsan_project_server_exp" # run_server.sh has fl_tsan_project_server
export WANDB_RUN_GROUP_SERVER="federated_tsan_server_main_group"
# WANDB_RUN_NAME_SERVER is set by run_server.sh with fedcross prefix

# --- Client Configuration ---
TOTAL_CLIENTS_TO_START=4
CLIENT_SERVER_ADDRESS="127.0.0.1:8089" # Should match SERVER_ADDRESS
CLIENT_EXCEL_PATH="/home/canhdx/workspace/age-prediction-using-MRI/label.xlsx"
CLIENT_WANDB_PROJECT="fl_tsan_project_clients_exp"

# --- Paths ---
# INITIAL_MODEL_PATH is checked and used by run_server.sh
export INITIAL_MODEL_PATH="/home/canhdx/workspace/age-prediction-using-MRI/flower/saved_models/initial_model.pth"
VENV_ACTIVATE_PATH="/home/canhdx/miniconda3/envs/medical"

# --- Define NIfTI and NumPy Data Paths ---
RAW_NIFTI_PARENT_DIR="/home/canhdx/workspace/age-prediction-using-MRI/data_per_client"
PROCESSED_NUMPY_PARENT_DIR="/home/canhdx/workspace/age-prediction-using-MRI/data_per_client"


# --- Activate Virtual Environment ---
if [ -f "$VENV_ACTIVATE_PATH" ]; then
  echo "Activating Python virtual environment from $VENV_ACTIVATE_PATH..."
  source "$VENV_ACTIVATE_PATH"
else
  echo "WARNING: Virtual environment not found at $VENV_ACTIVATE_PATH. Assuming packages are globally available."
fi

# --- GPU Sanity Check for Clients ---
# (This remains the same)
if [ "$TOTAL_CLIENTS_TO_START" -gt "$NUM_AVAILABLE_GPUS" ]; then
    echo "WARNING: TOTAL_CLIENTS_TO_START ($TOTAL_CLIENTS_TO_START) is greater than NUM_AVAILABLE_GPUS ($NUM_AVAILABLE_GPUS)."
    echo "Client GPU assignment might lead to errors or unwanted sharing if not handled carefully."
    echo "Proceeding with direct GPU ID assignment (0, 1, 2,...). Ensure this is intended."
fi

# --- COMMENTED OUT Original Split Data Section ---
echo "INFO: Assuming NIfTI data is already split into client folders under $RAW_NIFTI_PARENT_DIR."


# --- Preprocess NIfTI Data to NumPy for Clients (NEW SECTION) ---
echo "--- Preprocessing NIfTI data for $TOTAL_CLIENTS_TO_START clients (if needed) ---"

# PROCESSED_NUMPY_PARENT_DIR is set earlier in your script to:
# /home/canhdx/workspace/age-prediction-using-MRI/data_per_client
CLIENT_0_PROCESSED_NUMPY_PATH="$PROCESSED_NUMPY_PARENT_DIR/client_0_numpy"

# This is the check:
if [ -d "$CLIENT_0_PROCESSED_NUMPY_PATH" ] && [ "$(ls -A "$CLIENT_0_PROCESSED_NUMPY_PATH")" ]; then
    echo "INFO: Data for client 0 already appears to be preprocessed into NumPy format at $CLIENT_0_PROCESSED_NUMPY_PATH. Skipping preprocessing for all clients."
else
    # This block (calling preprocess_data.py) only runs if the check above fails
    echo "INFO: Running NIfTI to NumPy preprocessing..."
    for i in $(seq 0 $((TOTAL_CLIENTS_TO_START - 1)))
    do
        CURRENT_CLIENT_RAW_NIFTI_DIR="$RAW_NIFTI_PARENT_DIR/client_$i" # Input NIfTI
        echo "Preprocessing NIfTI data for client $i from $CURRENT_CLIENT_RAW_NIFTI_DIR to $PROCESSED_NUMPY_PARENT_DIR/client_${i}_numpy ..."
        
        python preprocess_data.py \
            --client-id=$i \
            --raw-nifti-input-dir="$CURRENT_CLIENT_RAW_NIFTI_DIR" \
            --processed-numpy-output-parent-dir="$PROCESSED_NUMPY_PARENT_DIR"
    done
    echo "INFO: NIfTI to NumPy preprocessing complete."
fi

# --- Start Server in Background using run_server.sh ---
echo "INFO: Starting Flower Server via ./run_server.sh"
# Ensure run_server.sh is executable: chmod +x run_server.sh
# Also ensure run_server.sh directs its output to server.log if you want that behavior.
# If run_server.sh doesn't background itself, you'll need to add '&' here.
# Assuming run_server.sh runs python server.py in the foreground:
./run_server.sh > server.log 2>&1 &
SERVER_PID=$! # Get the PID of the backgrounded run_server.sh script
echo "INFO: Server (via run_server.sh) PID: $SERVER_PID. Logs should be in server.log (if run_server.sh directs output)."

# --- Wait for Server and Check ---
SERVER_GRPC_INIT_WAIT_TIME=10
echo "INFO: Waiting $SERVER_GRPC_INIT_WAIT_TIME seconds for server's gRPC to initialize..."
sleep $SERVER_GRPC_INIT_WAIT_TIME

# Check if server process (run_server.sh or python server.py started by it) is running
# And if the port is listening
if ps -p $SERVER_PID > /dev/null; then # Checks if run_server.sh process is running
    echo "INFO: Server script (run_server.sh PID $SERVER_PID) is running."
    # Check if the actual server (python server.py) is listening on the port
    # The SERVER_ADDRESS variable might be set in run_server.sh, use that or the one from this script
    SERVER_PORT_TO_CHECK=$(echo $SERVER_ADDRESS | cut -d':' -f2) # Extract port
    if ss -tuln | grep -q ":${SERVER_PORT_TO_CHECK}.*LISTEN"; then
        echo "INFO: Server appears to be listening on port ${SERVER_PORT_TO_CHECK}."
    else
        echo "WARNING: Server script is running BUT python server.py may not be listening on port ${SERVER_PORT_TO_CHECK} after $SERVER_GRPC_INIT_WAIT_TIME seconds."
        echo "Check server.log (and any logs from run_server.sh) for errors. Clients might fail to connect."
    fi
else
    echo "ERROR: Server script (run_server.sh PID $SERVER_PID) is NOT running after $SERVER_GRPC_INIT_WAIT_TIME seconds!"
    echo "Check server.log (and any logs from run_server.sh) for early crash messages."
    exit 1
fi


# --- Start Clients Sequentially, Assigning GPUs ---
# (This section with CLIENT_GPU_MAP=(0 1 2 4) remains the same as per your last request)
CLIENT_PIDS=()
declare -a CLIENT_GPU_MAP=(0 1 2 3) # Your custom GPU mapping

if [ "$TOTAL_CLIENTS_TO_START" -ne "${#CLIENT_GPU_MAP[@]}" ]; then
    echo "ERROR: TOTAL_CLIENTS_TO_START ($TOTAL_CLIENTS_TO_START) does not match the number of GPUs defined in CLIENT_GPU_MAP (${#CLIENT_GPU_MAP[@]})."
    exit 1
fi

for i in $(seq 0 $((TOTAL_CLIENTS_TO_START - 1)))
do
  GPU_ID_FOR_CLIENT=${CLIENT_GPU_MAP[$i]}
  if [ "$GPU_ID_FOR_CLIENT" -ge "$NUM_AVAILABLE_GPUS" ]; then
      echo "ERROR: Mapped GPU ID $GPU_ID_FOR_CLIENT for client $i is not available (NUM_AVAILABLE_GPUS: $NUM_AVAILABLE_GPUS)."
      exit 1
  fi

  echo "-----------------------------------------------------"
  echo "INFO: Preparing to launch Client $i on physical GPU $GPU_ID_FOR_CLIENT (Custom Mapping)"
  export CUDA_VISIBLE_DEVICES=$GPU_ID_FOR_CLIENT
  echo "INFO: Launching Client $i (CID: $i, Physical GPU: $GPU_ID_FOR_CLIENT) via ./run_client.sh"
  export WANDB_RUN_NAME="client_${i}_${FL_STRATEGY}_gpu${GPU_ID_FOR_CLIENT}_run_$(date +%Y%m%d_%H%M%S)"
  ./run_client.sh $i $TOTAL_CLIENTS_TO_START "$CLIENT_SERVER_ADDRESS" > "client_${i}_gpu${GPU_ID_FOR_CLIENT}.log" 2>&1 &
  CLIENT_PIDS+=($!)
  echo "INFO: Client $i launched with PID ${CLIENT_PIDS[$i]}. Logs in client_${i}_gpu${GPU_ID_FOR_CLIENT}.log."
  if [ $i -lt $((TOTAL_CLIENTS_TO_START - 1)) ]; then
    sleep 5
  fi
done

# --- (Rest of the script: cleanup, wait $SERVER_PID, etc. remains the same) ---
echo "-----------------------------------------------------"
echo "INFO: All $TOTAL_CLIENTS_TO_START client processes launched."
echo "INFO: Server (PID $SERVER_PID via run_server.sh) is running with strategy $FL_STRATEGY (or strategy from run_server.sh)."
echo "INFO: Client PIDs: ${CLIENT_PIDS[*]}"
echo "INFO: Monitor server.log and client_X_gpuY.log files."
echo "INFO: Press Ctrl+C to attempt to stop the server and then clients."

cleanup() {
    echo -e "\n--- Cleaning up ---"
    for pid in "${CLIENT_PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            echo "INFO: Stopping client (PID $pid)..."
            kill $pid
        fi
    done
    sleep 1

    if ps -p $SERVER_PID > /dev/null; then # This is PID of run_server.sh
        echo "INFO: Stopping server script (PID $SERVER_PID)..."
        kill $SERVER_PID # This will kill run_server.sh; ensure server.py also terminates
        # If server.py is a child process of run_server.sh, it might get SIGTERM too.
        # If not, you might need a more robust way to find and kill server.py if run_server.sh doesn't clean it up.
    else
        echo "INFO: Server script (PID $SERVER_PID) was already stopped."
    fi
    wait
    echo "INFO: Cleanup attempt complete."
}

trap cleanup SIGINT SIGTERM

wait $SERVER_PID
SERVER_EXIT_CODE=$?
echo "INFO: Server script (PID $SERVER_PID) has ended with exit code $SERVER_EXIT_CODE."
cleanup