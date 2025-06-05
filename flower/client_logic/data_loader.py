import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MRIDataset(Dataset):
    def __init__(self, subject_ids, labels_df, data_dir, id_col='subject_id', label_col='subject_age', gender_col='subject_sex', use_gender_flag=True):
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        self.label_col = label_col
        self.gender_col = gender_col
        self.use_gender_flag = use_gender_flag
        self.labels_df = labels_df

        self.valid_subject_ids_in_dataset = []
        self.labels_in_dataset = []
        self.gender_in_dataset = []

        for subject_id in self.subject_ids:
            subject_id_str = str(subject_id)
            file_path = os.path.join(self.data_dir, f"{subject_id_str}.npy")
            if os.path.exists(file_path):
                try:
                    subject_data = self.labels_df.loc[subject_id_str]
                    label = float(subject_data[self.label_col])
                    self.labels_in_dataset.append(label)
                    if self.use_gender_flag:
                        gender_val = int(subject_data[self.gender_col])
                        gender_one_hot = torch.zeros(2, dtype=torch.float32)
                        gender_one_hot[gender_val] = 1.0
                        self.gender_in_dataset.append(gender_one_hot)
                    else:
                        self.gender_in_dataset.append(torch.empty(0, dtype=torch.float32))
                    self.valid_subject_ids_in_dataset.append(subject_id_str)
                except KeyError:
                    logger.warning(f"Label data for subject ID {subject_id_str} not found in provided labels_df. Skipping.")
                except ValueError as e:
                    logger.warning(f"Could not convert label or gender for subject ID {subject_id_str}: {e}. Skipping.")
                except Exception as e:
                    logger.warning(f"Error processing subject {subject_id_str} label/gender data: {e}. Skipping.")
            else:
                logger.warning(f"Data file not found for subject ID {subject_id_str} at {file_path}. Skipping.")

        if not self.valid_subject_ids_in_dataset:
            logger.warning("No valid data found for this dataset split after checks (files and labels).")
        if self.labels_in_dataset:
            self.labels_in_dataset = torch.tensor(self.labels_in_dataset, dtype=torch.float32)
        else:
            self.labels_in_dataset = torch.empty(0, dtype=torch.float32)
        if self.use_gender_flag and self.gender_in_dataset:
            if any(t.numel() > 0 for t in self.gender_in_dataset):
                try:
                    self.gender_in_dataset = torch.stack(self.gender_in_dataset)
                except RuntimeError as e:
                    logger.error(f"Error stacking gender tensors: {e}. This might indicate inconsistent gender data processing.")
                    if all(t.numel() == 0 for t in self.gender_in_dataset):
                        self.gender_in_dataset = torch.empty((len(self.gender_in_dataset),0), dtype=torch.float32)
                    else:
                         logger.error("Mixed empty and non-empty gender tensors, cannot stack. Gender data will be problematic.")
                         self.gender_in_dataset = [torch.zeros(2, dtype=torch.float32) if t.numel() > 0 else torch.empty(0, dtype=torch.float32) for t in self.gender_in_dataset]
        elif not self.use_gender_flag:
             self.gender_in_dataset = [torch.empty(0, dtype=torch.float32)] * len(self.valid_subject_ids_in_dataset)

    def __len__(self):
        return len(self.valid_subject_ids_in_dataset)

    def __getitem__(self, idx):
        subject_id = self.valid_subject_ids_in_dataset[idx]
        file_path = os.path.join(self.data_dir, f"{subject_id}.npy")
        try:
            image = np.load(file_path)
            image_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label_tensor = self.labels_in_dataset[idx].unsqueeze(0)
            gender_tensor = self.gender_in_dataset[idx]
            return image_tensor, gender_tensor, label_tensor
        except Exception as e:
            logger.error(f"Error loading data for subject {subject_id} in __getitem__: {e}. Returning dummy data.")
            dummy_image = torch.zeros((1, 91, 109, 91), dtype=torch.float32)
            dummy_label = torch.tensor([-1.0], dtype=torch.float32)
            dummy_gender = torch.empty(0, dtype=torch.float32)
            if self.use_gender_flag:
                dummy_gender = torch.zeros(2, dtype=torch.float32)
            return dummy_image, dummy_gender, dummy_label

def get_data_loaders(excel_path, client_data_dir, client_id_num, train_split_ratio=0.7, val_split_ratio=0.15, batch_size=16,
                     id_col='subject_id', label_col='subject_age', gender_col='subject_sex',
                     use_gender_config=True, random_seed=42):
    logger.info(f"--- Data Loading for Client {client_id_num} from preprocessed directory: {client_data_dir} ---")
    if not os.path.isdir(client_data_dir):
        logger.error(f"Client data directory not found: {client_data_dir}")
        return None, None, 0
    try:
        df_all_labels = pd.read_excel(excel_path)
        if id_col not in df_all_labels.columns:
            logger.error(f"ID column '{id_col}' not found in {excel_path}.")
            return None, None, 0
        if label_col not in df_all_labels.columns:
            logger.error(f"Label column '{label_col}' not found in {excel_path}.")
            return None, None, 0
        if use_gender_config and gender_col not in df_all_labels.columns:
            logger.error(f"Gender column '{gender_col}' not found in {excel_path} but use_gender is True.")
            return None, None, 0
        df_all_labels[id_col] = df_all_labels[id_col].astype(str)
        df_all_labels[label_col] = pd.to_numeric(df_all_labels[label_col], errors='coerce')
        df_all_labels.dropna(subset=[label_col], inplace=True)
    except Exception as e:
        logger.error(f"Error reading or processing Excel file {excel_path}: {e}")
        return None, None, 0
    available_npy_subject_ids = []
    for f_name in os.listdir(client_data_dir):
        if f_name.endswith(".npy"):
            subject_id_from_file = f_name.split('.')[0]
            available_npy_subject_ids.append(subject_id_from_file)
    if not available_npy_subject_ids:
        logger.warning(f"No .npy files found in client data directory: {client_data_dir}")
        return None, None, 0
    logger.info(f"Found {len(available_npy_subject_ids)} .npy files in {client_data_dir}.")
    client_specific_df = df_all_labels[df_all_labels[id_col].isin(available_npy_subject_ids)].copy()
    if client_specific_df.empty:
        logger.warning(f"Client {client_id_num} has 0 subjects after matching .npy files with labels from {excel_path}.")
        return None, None, 0
    logger.info(f"Client {client_id_num} has {len(client_specific_df)} subjects with both .npy data and labels.")
    try:
        if client_specific_df[id_col].duplicated().any():
            logger.warning(f"Duplicate subject IDs found in Excel for subjects present in {client_data_dir}. Keeping first occurrence.")
            client_specific_df.drop_duplicates(subset=[id_col], keep='first', inplace=True)
        client_specific_df_indexed = client_specific_df.set_index(id_col)
    except KeyError:
        logger.error(f"Failed to set index on client_specific_df using '{id_col}'. This is unexpected.")
        return None, None, 0
    subject_ids_for_split = client_specific_df_indexed.index.tolist()
    num_total_subjects = len(subject_ids_for_split)
    train_ids, val_ids = [], []
    tsr = max(0.0, min(1.0, train_split_ratio))
    vsr = max(0.0, min(1.0, val_split_ratio))
    if tsr + vsr > 1.0:
        logger.warning(f"train_split_ratio ({tsr}) + val_split_ratio ({vsr}) > 1.0 for client {client_id_num}. Scaling them down to sum to approx 1.0 while maintaining ratio.")
        current_sum = tsr + vsr
        tsr = tsr / current_sum
        vsr = vsr / current_sum
    num_train = int(round(num_total_subjects * tsr))
    num_val = int(round(num_total_subjects * vsr))
    if num_train + num_val > num_total_subjects:
        if num_train > num_total_subjects:
            num_train = num_total_subjects
            num_val = 0
        else:
            num_val = num_total_subjects - num_train
    if num_train < 0: num_train = 0
    if num_val < 0: num_val = 0
    if num_total_subjects == 0:
        pass
    elif num_total_subjects == 1:
        logger.info(f"Client {client_id_num} has only 1 sample. Assigning to train.")
        train_ids = subject_ids_for_split
    elif num_train == 0 and num_val == 0:
        logger.warning(f"Client {client_id_num} calculated 0 train and 0 val samples based on ratios. Assigning all {num_total_subjects} to train.")
        train_ids = subject_ids_for_split
    elif num_train == 0:
        logger.info(f"Client {client_id_num} has 0 train samples after split. Assigning all {num_total_subjects} to validation.")
        val_ids = subject_ids_for_split
    elif num_val == 0:
        logger.info(f"Client {client_id_num} has 0 validation samples after split. Assigning all {num_total_subjects} to train.")
        train_ids = subject_ids_for_split
    else:
        num_train_val = num_train + num_val
        if num_train_val == num_total_subjects:
            train_val_ids = subject_ids_for_split
        else:
            train_val_ids, _ = train_test_split(
                subject_ids_for_split,
                train_size=num_train_val,
                random_state=random_seed,
                shuffle=True,
            )
        if num_train == len(train_val_ids):
             train_ids = train_val_ids
        elif num_val == len(train_val_ids):
            val_ids = train_val_ids
        else:
            train_ids, val_ids = train_test_split(
                train_val_ids,
                train_size=num_train,
                test_size=num_val,
                random_state=random_seed,
                shuffle=True,
            )
    logger.info(f"Client {client_id_num} - Train size: {len(train_ids)}, Val size: {len(val_ids)}")
    train_dataset = MRIDataset(train_ids, client_specific_df_indexed, client_data_dir, id_col, label_col, gender_col, use_gender_config) if train_ids else None
    val_dataset = MRIDataset(val_ids, client_specific_df_indexed, client_data_dir, id_col, label_col, gender_col, use_gender_config) if val_ids else None
    num_actual_train_samples = len(train_dataset) if train_dataset and len(train_dataset.valid_subject_ids_in_dataset) > 0 else 0
    num_actual_val_samples = len(val_dataset) if val_dataset and len(val_dataset.valid_subject_ids_in_dataset) > 0 else 0
    if num_actual_train_samples != len(train_ids):
        logger.warning(f"Initial train_ids count was {len(train_ids)}, but MRIDataset found {num_actual_train_samples} valid train samples.")
    if num_actual_val_samples != len(val_ids):
         logger.warning(f"Initial val_ids count was {len(val_ids)}, but MRIDataset found {num_actual_val_samples} valid val samples.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True) if num_actual_train_samples > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if num_actual_val_samples > 0 else None
    if num_actual_train_samples == 0:
        logger.warning(f"Client {client_id_num} ended up with 0 training samples after all checks. No training will occur for this client.")
        return None, None, 0
    return train_loader, val_loader, num_actual_train_samples
    