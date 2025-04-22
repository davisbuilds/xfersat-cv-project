# Comparing Fine-Tuned CNNs on Satellite Imagery

Investigating transfer learning approaches on CNN model architectures for satellite imagery classification.

## Project Overview

This project evaluates transfer learning techniques for satellite imagery classification. It implements convolutional neural networks (ResNet and EfficientNet architectures) pre-trained on ImageNet and fine-tuned on the EuroSAT dataset, which contains 27,000 labeled RGB and multi-spectral satellite images across 10 classes of land use and land cover. Each class contains 2,000-3,000 geo-referenced images measuring 64×64 pixels with a spatial resolution of 10 meters per pixel. Results demonstrate that full fine-tuning significantly outperforms the frozen backbone approach across all models, highlighting the domain gap between natural images and satellite imagery, and suggesting that features learned on ImageNet require considerable adaptation to transfer effectively to remote sensing applications.

Frozen backbone model test accuracies:

| Model | Accuracy (test) |
| -------- | ------ |
| ResNet18 | 0.8232 |
| ResNet50 | 0.8637 |
| EfficientNet_B0 | 0.8074 |
| EfficientNet_B4 | 0.7452 |

Unfrozen backbone model test accuracies:

| Model | Accuracy (test) |
| -------- | ------ |
| ResNet18 | 0.9664 |
| ResNet50 | 0.9753 |
| EfficientNet_B0 | 0.9827 |
| EfficientNet_B4 | 0.9783 |

## Features

- Dataset exploration and visualization
- Data preparation with augmentation
- Transfer learning with multiple model architectures
- Model evaluation with confusion matrices
- Model interpretability with GradCAM visualization
- Misclassification analysis

## Setup

1. Clone this repository
2. Create a virtual environment: `uv venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies:
   - Using uv: `uv pip install -e .`
   - Using pip: `pip install -r requirements.txt`
5. Download and extract the EuroSAT dataset: `python src/data/make_dataset.py`
6. Or download manually from <https://github.com/phelber/eurosat> and place in `data/raw/EuroSAT_RGB/`

## Project Structure

- `data/` - Dataset storage
  - `raw/` - Original EuroSAT dataset
  - `processed/` - Processed data splits and mappings
- `models/` - Saved model checkpoints
- `notebooks/` - Jupyter notebooks:
  - `01_exploratory_analysis.ipynb` - Dataset exploration
  - `02_data_preparation.ipynb` - Data preparation and loading
  - `03_model_development.ipynb` - Model training and evaluation
  - `04_evaluation_visualization.ipynb` - Advanced visualization and analysis
  - `05_backbone_comparison.ipynb` - Comparison of frozen vs unfrozen backbones
  - `06_results_comparison.ipynb` - Visualization and comparison of model results
  - `07_progressive_unfreezing.ipynb` - Comparing transfer learning approaches (WIP)
- `src/` - Source code
  - `data/` - Data loading and processing
  - `models/` - Model definition and training
  - `visualization/` - Visualization utilities
- `main.py` - Command-line script for training models
- `run.py` - Simple script for training with default configuration
- `pyproject.toml` - Project configuration and dependencies
- `requirements.txt` - List of dependencies for pip installation
- `uv.lock` - Lock file for deterministic dependency resolution

## Usage

You can run the project in two ways:

### 1. Using notebooks

Run the notebooks in sequence:

1. First, explore the dataset: `jupyter lab notebooks/01_exploratory_analysis.ipynb`
2. Prepare the data: `jupyter lab notebooks/02_data_preparation.ipynb`
3. Train the models: `jupyter lab notebooks/03_model_development.ipynb`
4. Analyze results: `jupyter lab notebooks/04_evaluation_visualization.ipynb`

### 2. Using command-line scripts

- Train a model with default parameters: `python run.py`
- Train a model with custom parameters: `python main.py --model_name resnet50 --epochs 15 --batch_size 16 --freeze_backbone`

Available options for `main.py`:

- `--model_name`: Model architecture to use (resnet18, resnet50, efficientnet_b0, efficientnet_b4)
- `--freeze_backbone`: Freeze backbone weights
- `--batch_size`: Batch size for training
- `--epochs`: Number of epochs to train for
- `--learning_rate`: Learning rate
- `--num_workers`: Number of workers for data loading
- `--no_cuda`: Disable CUDA

## Key Dependencies

- PyTorch & torchvision - Deep learning framework
- scikit-learn - Evaluation metrics
- torchcam - Class activation map visualization
- matplotlib & seaborn - Visualization
