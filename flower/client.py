import re
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np
import os
import argparse
import wandb
import logging

from tsan_model.ScaleDense import ScaleDense
from tsan_model.ranking_loss import rank_difference_loss
from client_logic.data_loader import get_data_loaders
from utils import get_parameters_from_model, set_parameters_in_model

logger = logging.getLogger("FlowerClient")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

USE_GENDER_CONFIG = os.environ.get("USE_GENDER", "True").lower() == "true"
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.0001))
EPOCHS_PER_ROUND = int(os.environ.get("EPOCHS_PER_ROUND", 5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))
MAIN_LOSS_TYPE = os.environ.get("MAIN_LOSS_TYPE", "mse")
AUX_LOSS_TYPE = os.environ.get("AUX_LOSS_TYPE", "ranking")
LBD_AUX_LOSS = float(os.environ.get("LBD_AUX_LOSS", 10.0))
BETA_RANKING_LOSS = float(os.environ.get("BETA_RANKING_LOSS", 1.0))
SORTER_CHECKPOINT_PATH = os.environ.get("SORTER_CHECKPOINT_PATH", "/home/canhdx/workspace/age-prediction-using-MRI/flower/tsan_model/best_lstmla_slen_8.pth.tar")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Client (ID from script logic) starting. Effective PyTorch device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"Client mapped to physical GPU: {torch.cuda.get_device_name(0)}")

class TSANFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id_num, total_clients, data_dir, excel_path, wandb_project_name):
        self.client_id_num = client_id_num
        self.total_clients = total_clients
        self.data_dir = data_dir
        self.excel_path = excel_path
        self.wandb_project_name = wandb_project_name
        self.model = ScaleDense(nb_filter=8, nb_block=5, use_gender=USE_GENDER_CONFIG).to(DEVICE)
        self.train_loader, self.val_loader, self.num_train_examples = self._load_data()
        if MAIN_LOSS_TYPE.lower() == "mse":
            self.main_criterion = nn.MSELoss().to(DEVICE)
        elif MAIN_LOSS_TYPE.lower() == "mae":
            self.main_criterion = nn.L1Loss().to(DEVICE)
        else:
            logger.warning(f"Unsupported MAIN_LOSS_TYPE: {MAIN_LOSS_TYPE}. Defaulting to MSELoss.")
            self.main_criterion = nn.MSELoss().to(DEVICE)
        self.aux_criterion = None
        if AUX_LOSS_TYPE.lower() == "ranking" and LBD_AUX_LOSS > 0:
            if not os.path.exists(SORTER_CHECKPOINT_PATH):
                logger.error(f"Sorter checkpoint path for ranking loss not found: {SORTER_CHECKPOINT_PATH}. Ranking loss will be disabled.")
            else:
                try:
                    self.aux_criterion = rank_difference_loss(
                        sorter_checkpoint_path=SORTER_CHECKPOINT_PATH,
                        beta=BETA_RANKING_LOSS
                    ).to(DEVICE)
                    logger.info(f"Initialized ranking loss with sorter: {SORTER_CHECKPOINT_PATH}")
                except Exception as e:
                    logger.error(f"Failed to initialize rank_difference_loss: {e}. Ranking loss will be disabled.", exc_info=True)
                    self.aux_criterion = None
        elif AUX_LOSS_TYPE.lower() != "none":
            logger.warning(f"Unsupported AUX_LOSS_TYPE: {AUX_LOSS_TYPE} or LBD_AUX_LOSS is 0. Auxiliary loss disabled.")
        self.mae_metric_criterion = nn.L1Loss().to(DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        try:
            wandb.init(
                project=self.wandb_project_name,
                group="federated_tsan_clients_first_stage",
                name=os.environ.get("WANDB_RUN_NAME", f"client_{self.client_id_num}_tsan_fs_run"),
                config={
                    "client_id": self.client_id_num,
                    "total_clients": self.total_clients,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "epochs_per_round": EPOCHS_PER_ROUND,
                    "batch_size": BATCH_SIZE,
                    "use_gender": USE_GENDER_CONFIG,
                    "model": "ScaleDense_TSAN_FirstStage",
                    "data_dir": self.data_dir,
                    "main_loss": MAIN_LOSS_TYPE,
                    "aux_loss": AUX_LOSS_TYPE if self.aux_criterion else "none",
                    "lbd_aux_loss": LBD_AUX_LOSS if self.aux_criterion else 0,
                    "beta_ranking_loss": BETA_RANKING_LOSS if self.aux_criterion else 0,
                    "sorter_path": SORTER_CHECKPOINT_PATH if self.aux_criterion else "N/A"
                },
                reinit=True
            )
            logger.info(f"WandB initialized for client {self.client_id_num}")
        except Exception as e:
            logger.error(f"WandB initialization failed for client {self.client_id_num}: {e}")

    def _load_data(self):
        logger.info(f"Client {self.client_id_num}: Loading data from preprocessed NumPy directory: {self.data_dir}")
        train_loader, val_loader, num_train_examples = get_data_loaders(
            excel_path=self.excel_path,
            client_data_dir=self.data_dir,
            client_id_num=self.client_id_num,
            batch_size=BATCH_SIZE,
            train_split_ratio=0.7,
            val_split_ratio=0.15,
            use_gender_config=USE_GENDER_CONFIG,
            id_col=os.environ.get("SUBJECT_ID_COLUMN", "subject_id"),
            label_col=os.environ.get("LABEL_COLUMN", "subject_age"),
            gender_col=os.environ.get("GENDER_COLUMN", "subject_sex"),
            random_seed=42
        )
        if num_train_examples == 0 or train_loader is None:
            logger.error(f"Client {self.client_id_num} has no training data in {self.data_dir}. Exiting client.")
            raise ValueError(f"Client {self.client_id_num} could not load training data from {self.data_dir}.")
        logger.info(f"Client {self.client_id_num}: Data loaded. Training examples: {num_train_examples}")
        return train_loader, val_loader, num_train_examples

    def get_parameters(self, config):
        logger.info(f"Client {self.client_id_num}: Sending parameters to server.")
        return get_parameters_from_model(self.model)

    def set_parameters(self, parameters):
        logger.info(f"Client {self.client_id_num}: Receiving parameters from server.")
        set_parameters_in_model(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        current_round = config.get("server_round", 0)
        logger.info(f"Client {self.client_id_num}: Starting training for round {current_round}.")
        self.model.train()
        for epoch in range(EPOCHS_PER_ROUND):
            running_total_loss = 0.0
            running_main_loss = 0.0
            running_aux_loss = 0.0
            samples_processed = 0
            for images, gender_data, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                gender_data = gender_data.to(DEVICE)
                self.optimizer.zero_grad()
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
                model_gender_input = gender_data if USE_GENDER_CONFIG else None
                outputs = self.model(images, model_gender_input)   
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(1)
                loss1 = self.main_criterion(outputs, labels)
                loss2_val = 0.0
                if self.aux_criterion and LBD_AUX_LOSS > 0:
                    loss2 = self.aux_criterion(outputs, labels)
                    loss2_val = loss2.item()
                    total_loss = loss1 + LBD_AUX_LOSS * loss2
                else:
                    total_loss = loss1
                total_loss.backward()
                self.optimizer.step()
                running_total_loss += total_loss.item() * images.size(0)
                running_main_loss += loss1.item() * images.size(0)
                if self.aux_criterion and LBD_AUX_LOSS > 0:
                    running_aux_loss += loss2_val * images.size(0)
                samples_processed += images.size(0)
            epoch_total_loss = running_total_loss / samples_processed if samples_processed > 0 else 0
            epoch_main_loss = running_main_loss / samples_processed if samples_processed > 0 else 0
            epoch_aux_loss = running_aux_loss / samples_processed if samples_processed > 0 and self.aux_criterion and LBD_AUX_LOSS > 0 else 0
            log_message = (
                f"Client {self.client_id_num} - Round {current_round} - Epoch {epoch+1}/{EPOCHS_PER_ROUND} - "
                f"TotalLoss: {epoch_total_loss:.4f}, MainLoss: {epoch_main_loss:.4f}"
            )
            if self.aux_criterion and LBD_AUX_LOSS > 0:
                log_message += f", AuxLoss: {epoch_aux_loss:.4f}"
            logger.info(log_message)
            if wandb.run:
                log_dict = {
                    "round": current_round,
                    "epoch": epoch + 1,
                    f"client_{self.client_id_num}_train_total_loss_epoch": epoch_total_loss,
                    f"client_{self.client_id_num}_train_main_loss_epoch": epoch_main_loss,
                }
                if self.aux_criterion and LBD_AUX_LOSS > 0:
                     log_dict[f"client_{self.client_id_num}_train_aux_loss_epoch"] = epoch_aux_loss
                wandb.log(log_dict)
        logger.info(f"Client {self.client_id_num}: Finished training round {current_round}.")
        return get_parameters_from_model(self.model), self.num_train_examples, {"client_id": self.client_id_num, "round": current_round}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        current_round = config.get("server_round", 0)
        logger.info(f"Client {self.client_id_num}: Starting evaluation for round {current_round}.")
        if not self.val_loader:
            logger.warning(f"Client {self.client_id_num}: No validation loader. Returning zero metrics.")
            if wandb.run:
                wandb.log({"round": current_round, f"client_{self.client_id_num}_val_mae": float('inf'), f"client_{self.client_id_num}_val_loss": float('inf')})
            return float('inf'), 0, {"mae": float('inf'), "client_id": self.client_id_num, "round": current_round}
        self.model.eval()
        total_val_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, gender_data, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                gender_data = gender_data.to(DEVICE)
                if labels.ndim == 1:
                    labels = labels.unsqueeze(1)
                model_gender_input = gender_data if USE_GENDER_CONFIG else None
                outputs = self.model(images, model_gender_input)
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(1)
                loss = self.main_criterion(outputs, labels)
                mae = self.mae_metric_criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                total_mae += mae.item() * images.size(0)
                total_samples += images.size(0)
        avg_val_loss = total_val_loss / total_samples if total_samples > 0 else float('inf')
        avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
        logger.info(f"Client {self.client_id_num} - Round {current_round} - Validation Loss (Main): {avg_val_loss:.4f}, MAE: {avg_mae:.4f}")
        if wandb.run:
            wandb.log({
                "round": current_round,
                f"client_{self.client_id_num}_val_loss": avg_val_loss,
                f"client_{self.client_id_num}_val_mae": avg_mae
            })
        return avg_val_loss, total_samples, {"mae": avg_mae, "client_id": self.client_id_num, "round": current_round}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client for TSAN First Stage Training")
    parser.add_argument("--client-id-num", type=int, required=True, help="Numeric ID for this client (e.g., 0, 1, 2)")
    parser.add_argument("--total-clients", type=int, required=True, help="Total number of clients in the FL setup")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:808", help="Address of the Flower server")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing this client's MRI .npy files (e.g., .../client_0_numpy/)")
    parser.add_argument("--excel-path", type=str, required=True, help="Path to the Excel file with labels")
    parser.add_argument("--wandb-project", type=str, required=True, help="Weights & Biases project name for clients")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay for Adam optimizer")
    parser.add_argument("--main-loss-type", type=str, default=MAIN_LOSS_TYPE, choices=['mse', 'mae'], help="Main loss function type")
    parser.add_argument("--aux-loss-type", type=str, default=AUX_LOSS_TYPE, choices=['ranking', 'none'], help="Auxiliary loss function type")
    parser.add_argument("--lbd-aux-loss", type=float, default=LBD_AUX_LOSS, help="Lambda weight for auxiliary loss")
    parser.add_argument("--beta-ranking-loss", type=float, default=BETA_RANKING_LOSS, help="Beta parameter for ranking loss")
    parser.add_argument("--sorter-checkpoint-path", type=str, default=SORTER_CHECKPOINT_PATH, help="Path to SoDeep sorter checkpoint for ranking loss")
    parser.add_argument("--use-gender", type=str, default=str(USE_GENDER_CONFIG), help="Whether to use gender information ('True' or 'False')")
    args = parser.parse_args()
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    MAIN_LOSS_TYPE = args.main_loss_type
    AUX_LOSS_TYPE = args.aux_loss_type
    LBD_AUX_LOSS = args.lbd_aux_loss
    BETA_RANKING_LOSS = args.beta_ranking_loss
    SORTER_CHECKPOINT_PATH = args.sorter_checkpoint_path
    USE_GENDER_CONFIG = args.use_gender.lower() == "true"
    logger.info(f"--- Client {args.client_id_num} / {args.total_clients} ---")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Effective Configs:")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  Weight Decay: {WEIGHT_DECAY}")
    logger.info(f"  Use Gender: {USE_GENDER_CONFIG}")
    logger.info(f"  Main Loss: {MAIN_LOSS_TYPE}")
    logger.info(f"  Aux Loss: {AUX_LOSS_TYPE}, Lambda: {LBD_AUX_LOSS if AUX_LOSS_TYPE != 'none' else 'N/A'}")
    if AUX_LOSS_TYPE == 'ranking':
        logger.info(f"  Ranking Beta: {BETA_RANKING_LOSS}")
        logger.info(f"  Sorter Path: {SORTER_CHECKPOINT_PATH}")
    logger.info(f"Data directory for .npy files: {args.data_dir}")
    logger.info(f"Excel path: {args.excel_path}")
    logger.info(f"WandB Project: {args.wandb_project}")
    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        exit(1)
    if not os.path.isfile(args.excel_path):
        logger.error(f"Excel label file not found: {args.excel_path}")
        exit(1)
    if AUX_LOSS_TYPE == 'ranking' and not os.path.isfile(SORTER_CHECKPOINT_PATH):
        logger.warning(f"Sorter checkpoint path {SORTER_CHECKPOINT_PATH} not found. Ranking loss may be disabled if not found by the client class.")
    try:
        client_instance = TSANFlowerClient(
            client_id_num=args.client_id_num,
            total_clients=args.total_clients,
            data_dir=args.data_dir,
            excel_path=args.excel_path,
            wandb_project_name=args.wandb_project
        )
        fl.client.start_numpy_client(server_address=args.server_address, client=client_instance)
    except ValueError as e:
        logger.error(f"Failed to start client {args.client_id_num} due to ValueError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in client {args.client_id_num}: {e}", exc_info=True)
    finally:
        if wandb.run:
            wandb.finish()
            logger.info(f"WandB run finished for client {args.client_id_num}")