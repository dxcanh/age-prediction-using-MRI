import os
import argparse
import nibabel as nib
import numpy as np
import logging

from transform import (
    apply_rician_noise, apply_gaussian_blur, apply_synthetic_bias_field,
    apply_gamma_correction, resample, downsample_to_shape
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PreprocessData")

RICIAN_CONFIG = {
    'noise_std_range': (0.03, 0.05),
    'apply_prob': 0.8
}

def run_preprocessing(client_id_num, raw_nifti_input_dir, processed_numpy_output_parent_dir):
    logger.info(f"Starting preprocessing for client {client_id_num}")
    logger.info(f"Input NIfTI directory: {raw_nifti_input_dir}")

    output_dir_for_client_numpy = os.path.join(processed_numpy_output_parent_dir, f"client_{client_id_num}_numpy")
    os.makedirs(output_dir_for_client_numpy, exist_ok=True)
    logger.info(f"Output NumPy directory: {output_dir_for_client_numpy}")

    for file_name in os.listdir(raw_nifti_input_dir):
        if not (file_name.endswith(".gz") or file_name.endswith(".nii") or file_name.endswith(".nii.gz")):
            continue

        subject_id = file_name.split('_')[0] 
        
        logger.info(f"Processing file: {file_name} for subject ID: {subject_id}")
        subject_path = os.path.join(raw_nifti_input_dir, file_name)

        try:
            img = nib.load(subject_path)
            if not isinstance(img, nib.Nifti1Image):
                logger.error(f"File {file_name} is not a valid NIfTI image. Skipping.")
                continue

            data = img.get_fdata()
            pixdim = img.header['pixdim'][1:4]

            # Apply client-specific augmentations
            if client_id_num == 0:
                data = apply_gaussian_blur(data, pixdim, sigma_mm=0.8)
            elif client_id_num == 1:
                data = apply_gamma_correction(data, gamma=0.5)
            elif client_id_num == 2:
                data = apply_rician_noise(data, RICIAN_CONFIG) # RICIAN_CONFIG must be defined
            elif client_id_num == 3:
                data = apply_synthetic_bias_field(data, order=3)

            # Common transformations
            data = resample(data, pixdim, new_spacing=[2, 2, 2])
            data = downsample_to_shape(data, target_shape=(91, 109, 91))

            output_file_path = os.path.join(output_dir_for_client_numpy, subject_id + ".npy")
            np.save(output_file_path, data)
            logger.info(f"Saved processed data to {output_file_path}")

        except Exception as e:
            logger.error(f"Error processing file {file_name} for client {client_id_num}: {e}", exc_info=True)
    logger.info(f"Finished preprocessing for client {client_id_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NIfTI data to NumPy format for FL clients.")
    parser.add_argument("--client-id", type=int, required=True, help="Numeric ID for the client (e.g., 0, 1, 2, 3).")
    parser.add_argument("--raw-nifti-input-dir", type=str, required=True, help="Path to the client's raw NIfTI data directory (e.g., /path/to/data_per_client/client_0).")
    parser.add_argument("--processed-numpy-output-parent-dir", type=str, required=True, help="Parent directory where 'client_X_numpy' folders will be created (e.g., /path/to/skull_stripped).")

    args = parser.parse_args()

    run_preprocessing(
        client_id_num=args.client_id,
        raw_nifti_input_dir=args.raw_nifti_input_dir,
        processed_numpy_output_parent_dir=args.processed_numpy_output_parent_dir
    )