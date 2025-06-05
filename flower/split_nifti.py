import os
import shutil
import random
import glob

# --- Configuration ---
SOURCE_DIR = "/home/canhdx/workspace/age-prediction-using-MRI/skull_stripped"
BASE_DEST_DIR = "/home/canhdx/workspace/age-prediction-using-MRI/data_per_client"
NUM_CLIENTS = 4
NIFTI_EXTENSIONS = ("*.nii", "*.nii.gz") # Case-insensitive matching for glob
# If you need strict case sensitivity or more complex patterns, adjust accordingly.

# --- Helper Function ---
def get_nifti_files(source_directory, extensions):
    """
    Gets a list of all NIfTI files in the source directory.
    """
    all_files = []
    for ext in extensions:
        # glob is case-sensitive on Linux by default.
        # To make it effectively case-insensitive for common extensions:
        pattern_lower = ext.lower()
        pattern_upper = ext.upper()
        all_files.extend(glob.glob(os.path.join(source_directory, pattern_lower)))
        # Avoid adding duplicates if lower and upper are the same (e.g. '*.nii')
        if pattern_lower != pattern_upper:
            all_files.extend(glob.glob(os.path.join(source_directory, pattern_upper)))

    # Remove duplicates that might arise if a file matches both cases (though unlikely with NIfTI)
    # or if a file somehow matches multiple patterns (e.g. if extensions overlap)
    return sorted(list(set(all_files)))


# --- Main Script ---
if __name__ == "__main__":
    print(f"--- Starting NIfTI File Splitting Script ---")

    # 1. Check if source directory exists
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: Source directory '{SOURCE_DIR}' not found.")
        exit(1)

    # 2. Get all NIfTI files from the source directory
    print(f"Searching for NIfTI files in: {SOURCE_DIR}")
    nifti_files = get_nifti_files(SOURCE_DIR, NIFTI_EXTENSIONS)

    if not nifti_files:
        print(f"No NIfTI files found in '{SOURCE_DIR}' with extensions {NIFTI_EXTENSIONS}.")
        exit(0)

    print(f"Found {len(nifti_files)} NIfTI files.")

    # 3. Shuffle the files
    random.shuffle(nifti_files)
    print("Shuffled the file list randomly.")

    # 4. Create destination directories and prepare for distribution
    client_dirs = []
    for i in range(NUM_CLIENTS):
        client_dir_name = f"client_{i}"
        client_path = os.path.join(BASE_DEST_DIR, client_dir_name)
        client_dirs.append(client_path)
        os.makedirs(client_path, exist_ok=True) # exist_ok=True avoids error if dir exists
        print(f"Ensured destination directory exists: {client_path}")

    # 5. Distribute and copy files
    file_counts = [0] * NUM_CLIENTS
    print(f"\nCopying files to {NUM_CLIENTS} client directories...")

    for idx, src_file_path in enumerate(nifti_files):
        client_index = idx % NUM_CLIENTS
        dest_client_dir = client_dirs[client_index]
        
        file_name = os.path.basename(src_file_path)
        dest_file_path = os.path.join(dest_client_dir, file_name)

        try:
            # Check if file already exists in destination to avoid re-copying if script is re-run
            # and to allow for incremental additions if new files are added to source.
            # For a clean split, ensure destination folders are empty or remove this check.
            if os.path.exists(dest_file_path):
                print(f"Skipping '{file_name}', already exists in '{dest_client_dir}'.")
            else:
                print(f"Copying '{file_name}' to '{dest_client_dir}'...")
                shutil.copy2(src_file_path, dest_file_path) # copy2 preserves metadata
            file_counts[client_index] += 1
        except Exception as e:
            print(f"ERROR: Could not copy '{src_file_path}' to '{dest_file_path}': {e}")

    print("\n--- File Distribution Summary ---")
    total_copied_successfully = sum(file_counts) # This count might be higher if some files were skipped
    
    # To get a more accurate count of files *actually* copied in this run,
    # you'd need to track it separately from files already existing.
    # For simplicity, this summary shows files now present in each client dir attributed to this script's logic.

    for i in range(NUM_CLIENTS):
        print(f"Client {i} ({client_dirs[i]}): {file_counts[i]} files")
    
    print(f"\nTotal files processed for distribution: {len(nifti_files)}")
    print(f"Total files assigned to client directories (includes skipped pre-existing): {total_copied_successfully}")
    
    if len(nifti_files) == total_copied_successfully:
        print("All files were distributed successfully (or were already present).")
    else:
        print("Some files may not have been copied due to errors or pre-existence logic.")

    print("\n--- Script Finished ---")