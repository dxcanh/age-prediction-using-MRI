# Age Prediction Using MRI

A deep learning framework for brain age estimation from MRI scans using the TSAN (Two-Stage Attention Network) architecture with federated learning capabilities.

## Overview

This project implements a two-stage neural network approach for predicting brain age from MRI images. The framework supports both centralized training and federated learning scenarios, making it suitable for distributed medical data analysis while preserving privacy.

## Features

- **Two-Stage Architecture**: Implements TSAN with separate first and second stage training
- **Federated Learning**: Built-in support for distributed training across multiple clients
- **Data Preprocessing**: Comprehensive MRI data preprocessing pipeline
- **Multiple Loss Functions**: Support for MSE and other loss functions
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Experiment Tracking**: Integration with Weights & Biases (WandB)

## Project Structure

```
age-prediction-using-MRI/
├── data_preprocessing/          # Data preprocessing utilities
│   ├── preprocessing.py         # Main preprocessing script
│   ├── preprocessing.ipynb      # Jupyter notebook for preprocessing
│   └── split_data.py           # Data splitting utilities
├── flower/                      # Federated learning implementation
│   ├── client.py               # Federated client implementation
│   ├── server.py               # Federated server implementation
│   ├── split_nifti.py          # NIfTI file splitting for FL
│   └── requirements.txt        # FL-specific dependencies
├── TSAN/                       # TSAN model implementation
│   ├── train_first_stage.py    # First stage training
│   ├── train_second_stage.py   # Second stage training
│   └── utils/                  # Utility functions and configs
└── README.md
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- PyTorch
- NiBabel for NIfTI file handling

### Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

For federated learning:

```bash
cd flower
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

1. **Basic preprocessing**:
```bash
cd data_preprocessing
python preprocessing.py
```

2. **Data splitting**:
```bash
python split_data.py
```

### Centralized Training

#### First Stage Training

```bash
cd TSAN
python train_first_stage.py \
    --model_name first_stage_model.pth.tar \
    --output_dir ./model/ \
    --train_folder ../data/train \
    --valid_folder ../data/valid \
    --test_folder ../data/test \
    --excel_path ../labels/Training.xls \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001
```

#### Second Stage Training

```bash
python train_second_stage.py \
    --model_name second_stage_model.pth.tar \
    --first_stage_net ./model/first_stage_model.pth.tar \
    --output_dir ./model/ \
    --epochs 50 \
    --batch_size 16
```

### Federated Learning

#### Start the Server

```bash
cd flower
bash run_server.sh
```

#### Start Clients

```bash
# Terminal 1 (Client 0)
bash run_client.sh 0

# Terminal 2 (Client 1)  
bash run_client.sh 1

# Continue for additional clients...
```

## Configuration

Key configuration parameters in [`TSAN/utils/config.py`](TSAN/utils/config.py):

- `--model_name`: Checkpoint file name
- `--output_dir`: Output directory for models and logs
- `--train_folder`: Training data path
- `--valid_folder`: Validation data path  
- `--test_folder`: Test data path
- `--excel_path`: Labels file path
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--weight_decay`: L2 regularization weight
- `--use_gender`: Whether to use gender information
- `--loss`: Loss function type (default: 'mse')

## Model Architecture

The TSAN architecture consists of:

1. **First Stage**: Feature extraction and initial age prediction
2. **Second Stage**: Refinement network that takes first stage outputs and improves predictions

## Monitoring and Logging

The framework supports multiple monitoring options:

- **WandB Integration**: Automatic logging of metrics and visualizations
- **Local Logging**: Training logs saved to output directory
- **Early Stopping**: Configurable patience and validation metric monitoring

## Data Format

- **Input**: NIfTI (.nii.gz) MRI brain scans
- **Labels**: Excel file with age and metadata
- **Preprocessing**: Skull stripping and normalization

## Results

The model outputs:
- Mean Absolute Error (MAE) on test set
- Correlation Coefficient (CC) with ground truth ages
- Detailed training/validation metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is available for research and educational purposes. Please cite the relevant papers if you use this code in your research.

## Acknowledgments

- Based on TSAN architecture for brain age estimation
- Federated learning implementation using Flower framework
- NiBabel for NIfTI file processing
- PyTorch for deep learning framework

## Contact

For questions or issues, please open an issue in the repository or contact the maintainers.