# client/client.py
import requests
import numpy as np
import time
import os
import logging
from flask import Flask, jsonify, request
import random
import uuid
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# Import model and data loading functions
from model_pytorch import get_model, get_model_state_dict, set_model_state_dict
from data_loader import get_data_loaders

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine region/ID for logging and potentially data splitting
CLIENT_REGION = os.environ.get("REGION", "unknown")
# Attempt to get a unique numerical ID for data splitting (requires setup in start.sh or docker-compose)
# Defaulting to a random int if not set, but this isn't ideal for reproducible splits.
CLIENT_NUM_ID = int(os.environ.get("CLIENT_NUM_ID", random.randint(0, 1000)))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 2)) # Should match server's K

logger = logging.getLogger(f'FL-Client-{CLIENT_REGION}-{CLIENT_NUM_ID}')

app = Flask(__name__)
SERVER_URL = os.environ.get("SERVER_URL", "http://central-server:5000")
CLIENT_ID = f"{CLIENT_REGION}-{CLIENT_NUM_ID}-{uuid.uuid4().hex[:6]}" # Unique ID for server tracking

# Data Configuration - Use Environment Variables
DATA_DIR = os.environ.get("DATA_DIR", "/data/mri_numpy") # Path inside the container where .npy files are mounted
EXCEL_PATH = os.environ.get("EXCEL_PATH", "/data/labels/phenotypes.xlsx") # Path to the labels file
SUBJECT_ID_COLUMN = os.environ.get("SUBJECT_ID_COLUMN", "SubjectID") # Column name for subject IDs in Excel
LABEL_COLUMN = os.environ.get("LABEL_COLUMN", "Age") # Column name for age label in Excel

# Training Configuration
CLIENT_CONFIG = {
    "epochs_per_round": int(os.environ.get("EPOCHS_PER_ROUND", 3)), # Train for a few epochs locally
    "batch_size": int(os.environ.get("BATCH_SIZE", 16)), # Match TF code example
    "learning_rate": float(os.environ.get("LEARNING_RATE", 0.0001)), # Adjust as needed
    # Add other relevant configs if needed (e.g., optimizer type)
}

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Global Variables ---
model = get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=CLIENT_CONFIG['learning_rate'])
criterion = nn.MSELoss() # Mean Squared Error for age regression (could also use MAE/L1Loss)
mae_criterion = nn.L1Loss() # For validation loss reporting (MAE is often reported for age prediction)

train_loader, val_loader, test_loader, dataset_size = None, None, None, 0

# --- Helper Functions ---
def load_all_subject_ids(excel_path, id_col):
    """Loads all subject IDs from the main Excel file."""
    try:
        df = pd.read_excel(excel_path)
        # Ensure IDs are strings and handle potential loading issues
        if id_col not in df.columns:
             logger.error(f"Subject ID column '{id_col}' not found in {excel_path}")
             return []
        return df[id_col].astype(str).unique().tolist()
    except FileNotFoundError:
        logger.error(f"Master Excel file not found at {excel_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading master Excel file {excel_path}: {e}")
        return []

# --- Core Training and Evaluation Logic ---
def train_one_epoch(current_epoch):
    model.train()
    running_loss = 0.0
    samples_processed = 0

    if not train_loader:
        logger.warning("Training loader not available, skipping training epoch.")
        return 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        samples_processed += inputs.size(0)

        if (i + 1) % 20 == 0: # Log progress every 20 batches
             logger.info(f'Epoch [{current_epoch+1}/{CLIENT_CONFIG["epochs_per_round"]}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / samples_processed if samples_processed > 0 else 0.0
    logger.info(f"Epoch {current_epoch+1} Training Loss: {epoch_loss:.4f}")
    return epoch_loss


def evaluate_model():
    model.eval()
    total_mae_loss = 0.0
    total_samples = 0

    if not val_loader:
        logger.warning("Validation loader not available, returning high validation loss.")
        return float('inf') # Return infinity if no validation data

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            mae_loss = mae_criterion(outputs, labels) # Calculate MAE for validation
            total_mae_loss += mae_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_mae_loss = total_mae_loss / total_samples if total_samples > 0 else float('inf')
    logger.info(f"Validation MAE: {avg_mae_loss:.4f}")
    return avg_mae_loss


def run_federated_client():
    global model, train_loader, val_loader, test_loader, dataset_size, optimizer

    # --- Initial Data Loading ---
    logger.info("Attempting to load data...")
    all_subject_ids = load_all_subject_ids(EXCEL_PATH, SUBJECT_ID_COLUMN)
    if not all_subject_ids:
        logger.error("Failed to load subject IDs. Cannot proceed.")
        return

    # Get data loaders specific to this client
    train_loader, val_loader, test_loader, dataset_size = get_data_loaders(
        excel_path=EXCEL_PATH,
        data_dir=DATA_DIR,
        all_subject_ids=all_subject_ids,
        client_id=CLIENT_NUM_ID,
        num_clients=NUM_CLIENTS,
        train_split=0.7,
        val_split=0.15,
        batch_size=CLIENT_CONFIG['batch_size'],
        id_col=SUBJECT_ID_COLUMN,
        label_col=LABEL_COLUMN
    )

    if dataset_size == 0:
        logger.error("Client has no training data. Exiting.")
        return
    logger.info(f"Data loaded successfully. Training set size: {dataset_size}")
    # --- End Data Loading ---


    current_round = 0
    max_retries = 5
    retry_delay = 10 # seconds

    while True: # Main federated learning loop
        retries = 0
        model_fetched = False
        middleware_id = -1

        # 1. Fetch model from server
        while retries < max_retries and not model_fetched:
            try:
                logger.info(f"Attempting to fetch model from server for round {current_round}...")
                response = requests.get(f"{SERVER_URL}/model", params={'client_id': CLIENT_ID}, timeout=60)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                data = response.json()

                if data.get("status") == "wait":
                    wait_time = data.get("retry_after", 30)
                    logger.info(f"Server busy, waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1 # Count wait as a retry attempt
                    continue # Retry fetching

                # Successfully got a model
                server_model_state = data["model"]
                server_config = data["config"]
                new_round = server_config["round"]
                middleware_id = server_config["middleware_id"]

                # If it's a new round from the server, reset local state if necessary
                if new_round != current_round:
                    logger.info(f"Starting new round {new_round} (Server Round). Middleware ID: {middleware_id}")
                    current_round = new_round
                    # Reset optimizer state if desired per round, or adjust LR based on server round
                    optimizer = optim.Adam(model.parameters(), lr=CLIENT_CONFIG['learning_rate'])


                logger.info(f"Received middleware model {middleware_id} for round {current_round}.")
                model = set_model_state_dict(model, server_model_state, device)
                logger.info("Model state loaded successfully.")
                model_fetched = True

            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out. Retrying ({retries+1}/{max_retries})...")
                retries += 1
                time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching model: {e}. Retrying ({retries+1}/{max_retries})...")
                retries += 1
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"An unexpected error occurred during model fetch: {e}")
                retries += 1 # Treat unexpected errors as retries too
                time.sleep(retry_delay)


        if not model_fetched:
            logger.error("Failed to fetch model after multiple retries. Exiting.")
            break # Exit the main loop

        # 2. Train the local model
        logger.info(f"Starting local training for {CLIENT_CONFIG['epochs_per_round']} epochs...")
        for epoch in range(CLIENT_CONFIG['epochs_per_round']):
            train_one_epoch(epoch)
        logger.info("Local training finished.")

        # 3. Evaluate the trained model
        logger.info("Evaluating model on local validation set...")
        validation_loss = evaluate_model() # Using MAE
        logger.info(f"Local validation MAE after training: {validation_loss:.4f}")

        # 4. Send update to server
        logger.info("Sending updated model to server...")
        updated_model_state = get_model_state_dict(model)

        update_payload = {
            "client_id": CLIENT_ID,
            "middleware_id": middleware_id, # Include the ID of the model we updated
            "model": updated_model_state,
            "dataset_size": dataset_size,
            "validation_loss": validation_loss
        }

        retries = 0
        update_sent = False
        while retries < max_retries and not update_sent:
             try:
                 response = requests.post(f"{SERVER_URL}/update", json=update_payload, timeout=120) # Longer timeout for upload
                 response.raise_for_status()
                 logger.info(f"Update sent successfully: {response.json()}")
                 update_sent = True

                 # Check for early stopping signal from server
                 if response.json().get("early_stopping", False):
                     logger.info("Server indicated early stopping. Shutting down client.")
                     return # Exit cleanly

             except requests.exceptions.Timeout:
                 logger.warning(f"Update request timed out. Retrying ({retries+1}/{max_retries})...")
                 retries += 1
                 time.sleep(retry_delay)
             except requests.exceptions.RequestException as e:
                 logger.error(f"Error sending update: {e}. Retrying ({retries+1}/{max_retries})...")
                 retries += 1
                 time.sleep(retry_delay)
             except Exception as e:
                 logger.error(f"An unexpected error occurred during update send: {e}")
                 retries += 1
                 time.sleep(retry_delay)

        if not update_sent:
            logger.error("Failed to send update after multiple retries. Attempting next round.")
            # Decide whether to break or continue to the next round fetch
            # Let's continue, maybe the server will recover

        # Optional: Add a delay between rounds
        time.sleep(5)


if __name__ == '__main__':
    logger.info(f"Starting FL Client {CLIENT_ID} in region {CLIENT_REGION}")
    logger.info(f"Connecting to Server: {SERVER_URL}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Excel Path: {EXCEL_PATH}")
    run_federated_client()
    logger.info("Federated client finished.")