# DiffusionVAR: Enhanced Visual Autoregressive Modeling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DiffusionVAR** is an diffusion implementation of Visual Autoregressive (VAR) modeling that combines the power of autoregressive generation with masked diffusion modeling. This implementation extends the original [VAR](https://github.com/FoundationVision/VAR) with image evaluation integration visualisation tools, and wandb experiment tracking.

## 🚀 Key Features!

- **Dual Model Support**: Train both VAR and DiffusionVAR models with unified codebase
- **Rich Visualizations**: Comprehensive tools for analyzing model behavior and generation quality
- **Experiment Tracking**: Full WandB integration for monitoring training progress
- **Flexible Configuration**: YAML-based configuration with command-line overrides
- **Efficient Training**: Flash Attention and XFormers support

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Installation

### 1. Install PyTorch and Dependencies

First, install PyTorch with CUDA support (adjust for your CUDA version):

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install other required packages:

```bash
pip install -r requirements.txt
```

### 2. Prepare ImageNet Dataset

Prepare your ImageNet dataset in the following structure:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### 3. Optional Performance Optimizations

For faster attention computation, install optional packages:

```bash
# Flash Attention (recommended)
pip install flash-attn

# XFormers (alternative)
pip install xformers
```

The code will automatically detect and use these optimizations if available.

### Training VAR Model

Train a standard VAR model:

```bash
python train.py --config config/var.yaml --data_path /path/to/imagenet
```

### Training DiffusionVAR Model

Train a DiffusionVAR model with diffusion masking:

```bash
python train.py --config config/diffusionvar.yaml --data_path /path/to/imagenet
```

### Configuration

Configure training via YAML files or command-line arguments:

**YAML Configuration:**
```yaml
# config/var.yaml
algo: 'var'
data_path: '/path/to/imagenet'
batch_size: 128
learning_rate: 0.0004
epochs: 250
```

**Command-line Override:**
```bash
python train.py --config configF/var.yaml --batch_size 64 --learning_rate 0.0002
```

## Visualization and Analysis

### Generate Images

Generate images using trained models:

```bash
python visualisations/generation.py --config config/var.yaml --n_images 1000 --n_display_images 16 --save_dir results
```

### Analyze VQVAE Tokenization

Visualize VQVAE encoding and tokenization:

```bash
python visualisations/generation.py --config configs/diffusionvar.yaml --n_images 10000 --n_display_images 36 --save_dir dvar --classes "1_3_7" --n_class_images 1000 --ep 150
```

### Visualize Masking Patterns

Analyze different masking strategies:

```bash
python visualisations/masks.py
python visualisations/schedule.py
```

### WandB Integration

Enable experiment tracking with WandB:

```yaml
# In your config file
wandb:
  enable: true
  project: "diffusion-var"
  run_id: "experiment_1"
  log_interval: 100
```

### Model-Specific Arguments

**VAR Model:**
- `--depth`: Transformer depth (default: 16)
- `--embed_dim`: Embedding dimension (default: 1024)
- `--num_heads`: Number of attention heads (default: 16)

**DiffusionVAR Model:**
- `--diffusion_steps`: Number of diffusion steps (default: 1000)
- `--masking_schedule`: Masking schedule type (cosine, linear, etc.)

## 📁 Project Structure

```
dvar/
├── models/                 # Model implementations
│   ├── dvar/               # DiffusionVAR model components
│   │   ├── dvar.py         # Main DiffusionVAR model
│   │   └── var.py          # VAR model (moved here)
│   ├── var/                # VAR model package (empty, var.py moved to dvar/)
│   ├── vqvae/              # VQVAE tokenization
│   │   ├── vqvae.py        # Main VQVAE implementation
│   │   └── basic_vae.py    # VAE building blocks
│   ├── helpers.py          # Utility functions
│   ├── quant.py            # Vector quantization
│   ├── schedule.py         # Diffusion masking schedule
│   ├── transformer.py      # Transformer backbone
│   └── basic_var.py        # VAR building blocks
├── utils/                  # Utility functions
│   ├── image_metrics.py    # Image quality metrics (LPIPS, FID, IS)
│   ├── wandb_setup.py      # WandB integration
│   ├── data.py             # Data loading utilities
│   ├── lr_control.py       # Learning rate scheduling
│   ├── arg_util.py         # Argument parsing
│   ├── misc.py             # Miscellaneous utilities
│   └── ...
├── configs/                # Configuration files
│   ├── var.yaml            # VAR training config
│   ├── diffusionvar.yaml   # DiffusionVAR training config
│   └── diffusionvardiag.yaml # DiffusionVAR diagonal config
├── visualisations/         # Visualization tools
│   ├── generation.py       # Image generation visualization
│   ├── vqvae.py           # VQVAE analysis and visualization
│   ├── schedule.py         # Diffusion schedule visualization
│   ├── masks.py           # Masking visualization
│   ├── masks/              # Generated mask visualizations
│   ├── schedule/           # Generated schedule plots
│   └── vqvae/              # Generated VQVAE analysis plots
├── checkpoints/            # Model checkpoints
│   ├── dvar-diag/          # DiffusionVAR diagonal checkpoints
│   ├── dvar-last/          # Latest DiffusionVAR checkpoints
│   └── ...
├── train.py               # Main training script
├── trainer.py             # Training utilities and loop
├── dist.py                # Distributed training utilities
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features

### Advanced Visualizations

- **Token Analysis**: Visualize VQVAE tokenization patterns
- **Masking Visualization**: Analyze different masking strategies
- **Generation Quality**: Comprehensive image quality metrics
- **Training Monitoring**: Real-time loss and accuracy tracking

### Experiment Management

- **WandB Integration**: Automatic experiment tracking and logging
- **Checkpoint Management**: Automatic model saving and resuming
- **Configuration Management**: YAML-based config with validation
- **Reproducibility**: Fixed random seeds and deterministic training

## Acknowledgments

- [VAR](https://github.com/FoundationVision/VAR) - Original Visual Autoregressive Modeling implementation
- [MD4](https://github.com/darioShar/pytorch-md4) - Masked Diffusion 