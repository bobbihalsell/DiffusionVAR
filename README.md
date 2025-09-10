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
python visualisations/vqvae.py --config configF/var.yaml --data_path /path/to/imagenet
```

### Visualize Masking Patterns

Analyze different masking strategies:

```bash
python visualisations/masks.py
python visualisations/schedule.py
```

## 🔧 Configuration Options

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | Required |
| `--data_path` | Path to ImageNet dataset | Required |
| `--batch_size` | Training batch size | 128 |
| `--learning_rate` | Learning rate | 0.0004 |
| `--epochs` | Number of training epochs | 250 |
| `--device` | Device to use (cuda:0, cpu) | auto |
| `--num_workers` | DataLoader workers | 8 |

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
├── models/                 # Clean model implementations
│   ├── var/                # VAR model components
│   ├── dvar/               # DiffusionVAR model components
│   ├── vqvae/              # VQVAE tokenization
│   └── helpers.py          # Utility functions
├── utils/                  # Clean utility functions
│   ├── image_metrics.py    # Image quality metrics
│   ├── wandb_setup.py      # WandB integration
│   └── ...
├── config/                 # Clean configuration files
│   ├── var.yaml            # VAR training config
│   └── diffusionvar.yaml   # DiffusionVAR training config
├── visualisations/         # Visualization tools
│   ├── generation.py       # Image generation
│   ├── vqvae.py           # VQVAE analysis
│   └── ...
├── train.py               # Main training script
├── trainer.py             # Training utilities
└── requirements.txt        # Dependencies
```

## 🎯 Key Features Explained

### Enhanced Training Pipeline

- **Mixed Precision Training**: Automatic mixed precision with gradient scaling
- **Distributed Training**: Multi-GPU support with proper synchronization
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warmup

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

## 📈 Performance Tips

1. **Use Flash Attention**: Install `flash-attn` for 2-3x speedup
2. **Optimize Batch Size**: Use largest batch size that fits in GPU memory
3. **Enable Mixed Precision**: Reduces memory usage and speeds up training
4. **Use Multiple GPUs**: Scale training across multiple GPUs
5. **Monitor Memory**: Use `nvidia-smi` to monitor GPU memory usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [VAR](https://github.com/FoundationVision/VAR) - Original Visual Autoregressive Modeling implementation
- [DiT](https://github.com/facebookresearch/DiT) - Diffusion Transformer architecture
- [Taming Transformers](https://github.com/CompVis/taming-transformers) - VQVAE implementation