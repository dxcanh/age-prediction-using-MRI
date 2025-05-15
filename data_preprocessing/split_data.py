import os
import shutil
from pathlib import Path
import random

def split_and_copy_nii_files(source_dir, num_clients=4):
    """
    Split .nii.gz files into num_clients parts and copy to respective client folders.
    
    Args:
        source_dir (str): Path to source directory containing .nii.gz files
        num_clients (int): Number of clients (default: 4)
    """
    # Convert source_dir and des_path to Path objects
    source_path = Path(source_dir)
    des_path = Path("/home/canhdx/workspace/age-prediction-using-MRI/data_per_client")
    
    # Verify if source directory exists
    if not source_path.exists():
        print(f"Error: Directory {source_dir} does not exist")
        return
    
    # Get all .nii.gz files in source directory
    all_files = [f for f in source_path.iterdir() if f.is_file() and f.suffixes == ['.nii', '.gz']]
    
    # Check if there are any .nii.gz files
    if not all_files:
        print(f"Error: No .nii.gz files found in {source_dir}")
        return
    
    # Shuffle files to ensure random distribution
    random.shuffle(all_files)
    
    # Calculate number of files per client
    total_files = len(all_files)
    files_per_client = total_files // num_clients
    remainder = total_files % num_clients
    
    # Create client directories and distribute files
    file_index = 0
    for client_idx in range(num_clients):
        # Create new client directory
        client_dir = des_path / f'client_{client_idx}'
        client_dir.mkdir(exist_ok=True)
        
        # Determine number of files for this client
        num_files = files_per_client + (1 if client_idx < remainder else 0)
        
        # Copy files to client directory
        for _ in range(num_files):
            if file_index < total_files:
                source_file = all_files[file_index]
                dest_file = client_dir / source_file.name
                shutil.copy2(source_file, dest_file)
                print(f'Copied {source_file.name} to client_{client_idx}')
                file_index += 1
    
    print(f'\nDistribution complete!')
    print(f'Total .nii.gz files processed: {total_files}')
    for i in range(num_clients):
        client_dir = des_path / f'client_{i}'
        client_files = len([f for f in client_dir.iterdir() if f.suffixes == ['.nii', '.gz']])
        print(f'client_{i}: {client_files} files')

if __name__ == '__main__':
    # Specify the source directory
    source_directory = '/home/canhdx/workspace/age-prediction-using-MRI/skull_stripped'
    
    # Split and copy .nii.gz files to 4 client folders
    split_and_copy_nii_files(source_directory, num_clients=4)