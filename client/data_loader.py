import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__) # Use the logger configured in client.py

# MRIDataset class remains the same as before
class MRIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MRI .npy files and corresponding labels.
    """
    def __init__(self, subject_ids, labels_df, data_dir, id_col='subject_id', label_col='subject_age'):
        """
        Args:
            subject_ids (list): List of subject IDs for this dataset split.
            labels_df (pd.DataFrame): DataFrame containing labels, indexed by subject_id.
            data_dir (str): Directory where .npy files are stored.
            id_col (str): Column name for subject IDs in labels_df. THIS IS EXPECTED TO BE THE INDEX.
            label_col (str): Column name for the target label (e.g., age).
        """
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        # self.id_col = id_col # No longer needed if index is guaranteed
        self.label_col = label_col
        self.labels_df = labels_df # Expecting DataFrame already indexed by id_col

        self.valid_indices = []
        self.labels = []

        # Pre-filter labels and check file existence to avoid errors during training
        for idx, subject_id in enumerate(self.subject_ids):
            file_path = os.path.join(self.data_dir, f"{subject_id}.npy")
            if os.path.exists(file_path):
                try:
                    # Check if label exists for this subject using the index
                    label = self.labels_df.loc[subject_id, self.label_col]
                    # Attempt to convert label to float (should already be float after get_data_loaders processing)
                    float_label = float(label)
                    self.valid_indices.append(idx)
                    self.labels.append(float_label)
                except KeyError:
                    logger.warning(f"Label not found for subject ID {subject_id} in the provided DataFrame slice. Skipping.")
                except ValueError:
                     logger.warning(f"Could not convert label '{label}' to float for subject ID {subject_id} (This shouldn't happen here). Skipping.")
                except Exception as e:
                    logger.warning(f"Error processing label for subject ID {subject_id}: {e}. Skipping.")
            else:
                logger.warning(f"Data file not found for subject ID {subject_id} at {file_path}. Skipping.")

        if not self.valid_indices:
            logger.warning("No valid data found for this dataset split after checking files and labels.")

        self.labels = torch.tensor(self.labels, dtype=torch.float32) # Convert labels to float tensor


    def __len__(self):
        """Returns the number of valid samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Loads and returns a sample (image, label) from the dataset.
        """
        # Map the requested index to the index in the original subject_ids list
        original_list_idx = self.valid_indices[idx]
        subject_id = self.subject_ids[original_list_idx]
        file_path = os.path.join(self.data_dir, f"{subject_id}.npy")

        try:
            # Load the .npy file
            image = np.load(file_path)
            # Convert image to PyTorch tensor and add channel dimension (C, D, H, W) or (C, H, W, D)
            # Assuming input shape is (D, H, W) -> (65, 65, 55) or similar
            # Add channel dimension at the beginning: (1, D, H, W)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            # Get the pre-filtered and converted label
            label = self.labels[idx] # Get label corresponding to the filtered index

            # Ensure label is a tensor with a single element, suitable for regression loss
            if not isinstance(label, torch.Tensor):
                 label_tensor = torch.tensor([label], dtype=torch.float32)
            else:
                 label_tensor = label.unsqueeze(0) if label.ndim == 0 else label

            return image_tensor, label_tensor

        except FileNotFoundError:
            logger.error(f"File not found error during getitem for {subject_id} (should have been pre-checked). Returning dummy data.")
            dummy_image = torch.zeros((1, 65, 65, 55), dtype=torch.float32) # Match expected shape
            dummy_label = torch.tensor([-1.0], dtype=torch.float32)
            return dummy_image, dummy_label
        except Exception as e:
            logger.error(f"Error loading data for subject {subject_id}: {e}. Returning dummy data.")
            dummy_image = torch.zeros((1, 65, 65, 55), dtype=torch.float32)
            dummy_label = torch.tensor([-1.0], dtype=torch.float32)
            return dummy_image, dummy_label

# --- MODIFIED get_data_loaders function ---
def get_data_loaders(excel_path, data_dir, all_subject_ids, client_id, num_clients,
                     train_split=0.7, val_split=0.15, batch_size=16,
                     id_col='subject_id', label_col='subject_age', random_seed=42):
    """
    Loads label data, **assigns subjects to clients based on age ranges (Non-IID)**,
    splits the client's assigned data into train/val/test, creates datasets,
    and returns PyTorch DataLoaders.

    Args:
        excel_path (str): Path to the Excel file containing labels.
        data_dir (str): Path to the directory containing .npy image files.
        all_subject_ids (list): A list of all unique subject IDs considered for the FL process.
                                (Used for initial filtering before age splitting).
        client_id (int): The numerical index of the current client (0, 1, 2 mapped to age ranges).
        num_clients (int): The total number of clients (mainly for logging context, assignment is based on client_id).
        train_split (float): Proportion of the client's data for training.
        val_split (float): Proportion of the client's data for validation.
                           Test split will be 1 - train_split - val_split.
        batch_size (int): Batch size for the DataLoaders.
        id_col (str): Column name for subject IDs in the Excel file.
        label_col (str): Column name for the target label (age) in the Excel file.
        random_seed (int): Random seed for train/val/test splitting for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset_size)
               Returns (None, None, None, 0) if the client has no data or errors occur.
    """
    logger.info(f"--- Data Loading for Client {client_id} (Non-IID Age Split) ---")
    logger.info(f"Loading labels from: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
        # Basic validation
        if id_col not in df.columns:
            logger.error(f"Subject ID column '{id_col}' not found in {excel_path}.")
            return None, None, None, 0
        if label_col not in df.columns:
            logger.error(f"Label column (age) '{label_col}' not found in {excel_path}.")
            return None, None, None, 0
        # Convert subject IDs to string just in case they are loaded as numbers
        df[id_col] = df[id_col].astype(str)
        logger.info(f"Successfully loaded labels for {len(df)} total subjects from Excel.")

    except FileNotFoundError:
        logger.error(f"Excel file not found at {excel_path}")
        return None, None, None, 0
    except Exception as e:
        logger.error(f"Error reading Excel file {excel_path}: {e}")
        return None, None, None, 0

    # --- Initial Filtering based on all_subject_ids (Optional but recommended) ---
    if all_subject_ids:
        original_count_pre_filter = len(df)
        df = df[df[id_col].isin(all_subject_ids)].copy()
        logger.info(f"Filtered DataFrame from {original_count_pre_filter} to {len(df)} subjects based on provided 'all_subject_ids' list.")
        if len(df) == 0:
            logger.warning("No subjects remaining after filtering by 'all_subject_ids'. Cannot proceed.")
            return None, None, None, 0
    else:
        logger.warning("No 'all_subject_ids' list provided. Proceeding with all subjects found in the Excel file.")


    # --- Define Age Ranges and Select Client's Range ---
    # This mapping defines the Non-IID split based on client ID
    age_ranges = {
        0: (18, 30),  # Client ID 0 -> Ages 18-30
        1: (31, 55),  # Client ID 1 -> Ages 31-55
        2: (56, 90),  # Client ID 2 -> Ages 56-90
        # Add more ranges here if you increase num_clients and adjust client.py accordingly
    }

    if client_id not in age_ranges:
        # If using dynamic client IDs from environment variables, ensure they match 0, 1, 2...
        logger.error(f"Client ID {client_id} does not have a defined age range. Supported IDs for age splitting: {list(age_ranges.keys())}")
        return None, None, None, 0

    min_age, max_age = age_ranges[client_id]
    logger.info(f"Client {client_id} assigned age range: {min_age}-{max_age} (inclusive)")

    # --- Filter DataFrame by Assigned Age Range ---
    # Ensure the age column is numeric, converting non-numeric values to NaN
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')

    # Log and drop rows where age could not be converted or is missing
    original_count_age_filter = len(df)
    rows_with_nan_age = df[label_col].isna()
    if rows_with_nan_age.any():
        logger.warning(f"Found {rows_with_nan_age.sum()} rows with non-numeric or missing values in age column '{label_col}'. These rows will be excluded.")
        df.dropna(subset=[label_col], inplace=True)

    # Apply the age range filter
    client_df = df[(df[label_col] >= min_age) & (df[label_col] <= max_age)].copy()
    logger.info(f"Found {len(client_df)} subjects for client {client_id} within age range {min_age}-{max_age} (out of {original_count_age_filter} potential subjects before age filtering).")

    if len(client_df) == 0:
        logger.warning(f"Client {client_id} has 0 subjects matching the age range {min_age}-{max_age}. No data loaders will be created.")
        return None, None, None, 0

    # --- Prepare for Dataset Creation ---
    # Set the subject ID column as the index for efficient lookup in MRIDataset
    try:
        client_df.set_index(id_col, inplace=True)
    except KeyError:
        logger.error(f"Cannot set index. Column '{id_col}' might already be the index or doesn't exist after filtering.")
        return None, None, None, 0

    # Get the list of subject IDs assigned to this client (based on age)
    client_subject_ids_for_split = client_df.index.tolist()

    # --- Train/Val/Test Split for the Client's Data ---
    if len(client_subject_ids_for_split) < 3: # Need samples for train/val/test potentially
        logger.warning(f"Client {client_id} has fewer than 3 samples ({len(client_subject_ids_for_split)}) within the age range. Cannot perform reliable train/val/test split.")
        # Assign all available samples to training set
        if len(client_subject_ids_for_split) > 0:
             logger.warning("Assigning all samples to the training set.")
             train_ids = client_subject_ids_for_split
             val_ids = []
             test_ids = []
        else:
             return None, None, None, 0 # No samples at all
    else:
        # Calculate test split size
        test_split_prop = max(0.0, 1.0 - train_split - val_split)
        if test_split_prop < 0.001 and (train_split + val_split) >= 0.999: # Handle float precision
             test_split_prop = 0.0
             # logger.info("Train + Val split >= 1.0. Setting test split to 0.") # Less verbose
        elif (train_split + val_split) > 1.0:
             logger.error(f"Train split ({train_split}) + Val split ({val_split}) > 1.0. Invalid configuration.")
             return None, None, None, 0

        # First split: separate out the test set
        if test_split_prop > 0 and len(client_subject_ids_for_split) > 1 : # Need at least 2 samples to split
            try:
                train_val_ids, test_ids = train_test_split(
                    client_subject_ids_for_split,
                    test_size=test_split_prop,
                    random_state=random_seed,
                    stratify=None # Cannot stratify easily on continuous age within the client split
                )
            except ValueError as e:
                 logger.warning(f"Could not perform test split (might be due to small sample size). Assigning all to train/val. Error: {e}")
                 train_val_ids = client_subject_ids_for_split
                 test_ids = []

        else:
            train_val_ids = client_subject_ids_for_split
            test_ids = []

        # Second split: split the remaining data into train and validation
        if len(train_val_ids) > 0 and val_split > 0 and train_split > 0 :
             # Calculate validation proportion relative to the train+validation pool
             relative_val_split = val_split / (train_split + val_split)
             if len(train_val_ids) < 2: # Cannot split 1 sample
                  logger.warning("Only 1 sample remaining after test split. Assigning to train set.")
                  train_ids = train_val_ids
                  val_ids = []
             else:
                  try:
                    train_ids, val_ids = train_test_split(
                        train_val_ids,
                        test_size=relative_val_split,
                        random_state=random_seed, # Use the same seed for consistency
                        stratify=None
                    )
                  except ValueError as e:
                    logger.warning(f"Could not perform train/val split (might be due to small sample size). Assigning all to train set. Error: {e}")
                    train_ids = train_val_ids
                    val_ids = []

        elif len(train_val_ids) > 0 : # If either val_split or train_split is effectively 0 for the remainder
             train_ids = train_val_ids # Assign all remaining to train
             val_ids = []
        else: # train_val_ids is empty
             train_ids = []
             val_ids = []


    logger.info(f"Client {client_id} - Final Split Sizes -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # --- Create Datasets and DataLoaders ---
    # Pass the filtered and indexed client_df
    train_dataset = MRIDataset(train_ids, client_df, data_dir, label_col=label_col) # id_col is implicitly the index now
    val_dataset = MRIDataset(val_ids, client_df, data_dir, label_col=label_col)
    test_dataset = MRIDataset(test_ids, client_df, data_dir, label_col=label_col)

    # Create DataLoaders only if the corresponding dataset has samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if len(test_dataset) > 0 else None

    train_dataset_size = len(train_dataset)

    if train_dataset_size == 0 and (len(val_ids) > 0 or len(test_ids) > 0):
         logger.warning(f"Client {client_id} has 0 samples in the training set but samples exist in validation/test sets. Check split ratios and sample counts.")
    elif train_dataset_size == 0:
         logger.warning(f"Client {client_id} has 0 samples in the final training set after processing. Check data files, labels, and age range.")


    return train_loader, val_loader, test_loader, train_dataset_size
# --- End MODIFIED get_data_loaders function ---

# Example Usage (if __name__ == '__main__') remains the same, but will now test the age-based splitting
# if you run it standalone. You might want to adjust the dummy age data generation to span the ranges 18-90.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration for Testing ---
    DUMMY_DATA_DIR = './dummy_numpy_data'
    DUMMY_EXCEL_PATH = './dummy_labels.xlsx'
    N_SUBJECTS = 150 # Increase subjects for better range coverage
    N_CLIENTS_TEST = 3 # Match the number of age ranges
    IMG_SHAPE = (5, 5, 4)

    os.makedirs(DUMMY_DATA_DIR, exist_ok=True)

    # Create dummy excel with ages spanning the desired ranges
    subject_ids = [f"sub-Test{i:04d}" for i in range(N_SUBJECTS)]
    # Generate ages more deliberately to fall into the ranges
    ages = np.concatenate([
        np.random.uniform(18, 30, N_SUBJECTS // 3),
        np.random.uniform(31, 55, N_SUBJECTS // 3),
        np.random.uniform(56, 90, N_SUBJECTS - 2 * (N_SUBJECTS // 3))
    ]).round(1)
    np.random.shuffle(ages) # Shuffle ages after generation
    ages = ages[:N_SUBJECTS] # Ensure correct number if division wasn't exact

    sex = np.random.choice(['M', 'F'], N_SUBJECTS)
    # Add some potentially problematic entries
    subject_ids.append("sub-BadAgeText")
    ages = np.append(ages, "Unknown") # Non-numeric age
    sex = np.append(sex,"M")
    subject_ids.append("sub-NoAge")
    ages = np.append(ages, np.nan) # Missing age
    sex = np.append(sex,"F")


    dummy_df = pd.DataFrame({
        'subject_id': subject_ids,
        'subject_age': ages,
        'subject_sex': sex
    })
    dummy_df.to_excel(DUMMY_EXCEL_PATH, index=False)
    logger.info(f"Generated dummy labels in {DUMMY_EXCEL_PATH} with {len(dummy_df)} entries.")

    # Create dummy npy files only for subjects with valid-looking IDs in the final DataFrame
    all_ids_in_excel = dummy_df['subject_id'].tolist()
    created_files_count = 0
    for subj_id in all_ids_in_excel:
        # Simple check to avoid creating files for bad IDs if needed, although loading handles it
        if isinstance(subj_id, str) and subj_id.startswith("sub-"):
            dummy_img = np.random.rand(*IMG_SHAPE).astype(np.float32)
            np.save(os.path.join(DUMMY_DATA_DIR, f"{subj_id}.npy"), dummy_img)
            created_files_count += 1
    logger.info(f"Created {created_files_count} dummy .npy files in {DUMMY_DATA_DIR}")
    # --- End Dummy Data Creation ---


    # Test the get_data_loaders function for each client ID corresponding to an age range
    all_subject_ids_from_df = dummy_df['subject_id'].tolist() # Use all potential IDs for filtering test

    for c_id in range(N_CLIENTS_TEST): # Test client IDs 0, 1, 2
        logger.info(f"\n--- Testing Non-IID Loader for Client {c_id} ---")
        train_loader_test, val_loader_test, test_loader_test, train_size_test = get_data_loaders(
            excel_path=DUMMY_EXCEL_PATH,
            data_dir=DUMMY_DATA_DIR,
            all_subject_ids=all_subject_ids_from_df, # Pass all IDs generated
            client_id=c_id, # This now selects the age range
            num_clients=N_CLIENTS_TEST, # Provide context
            train_split=0.7,
            val_split=0.15,
            batch_size=4,
            id_col='subject_id',
            label_col='subject_age'
        )

        logger.info(f"Client {c_id} - Train Loader: {'Created' if train_loader_test else 'None'}, Val Loader: {'Created' if val_loader_test else 'None'}, Test Loader: {'Created' if test_loader_test else 'None'}")
        logger.info(f"Client {c_id} - Train Dataset Size: {train_size_test}")

        # Optional: Iterate through one batch to check shapes and maybe rough age range
        if train_loader_test and train_size_test > 0:
            try:
                images, labels = next(iter(train_loader_test))
                logger.info(f"Client {c_id} - Batch Image Shape: {images.shape}")
                logger.info(f"Client {c_id} - Batch Label Shape: {labels.shape}")
                logger.info(f"Client {c_id} - Batch Labels (Ages): {labels.squeeze().tolist()}") # Show sample ages
            except StopIteration:
                 logger.info(f"Client {c_id} - Train loader is empty (as expected if no data in range).")
            except Exception as e:
                 logger.error(f"Client {c_id} - Error iterating train_loader: {e}")

