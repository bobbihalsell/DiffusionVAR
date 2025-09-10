**DiffusionVAR** is an diffusion implementation of Visual Autoregressive (VAR) modeling that combines the power of autoregressive generation with masked diffusion modeling. This implementation extends the original [VAR](https://github.com/FoundationVision/VAR) with image evaluation integration visualisation tools, and wandb experiment tracking.


- **Dual Model Support**: Train both VAR and DiffusionVAR models with unified codebase
- **Rich Visualizations**: Comprehensive tools for analyzing model behavior and generation quality
- **Experiment Tracking**: Full WandB integration for monitoring training progress
- **Flexible Configuration**: YAML-based configuration with command-line overrides
- **Efficient Training**: Flash Attention and XFormers support


## Installation

### 1. Install Dependencies

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
python train.py --config config/var.yaml --batch_size 64 --learning_rate 0.0002
```

## Visualization and Analysis

### Generate Images

Generate images using trained models:

```bash
python visualisations/generation.py --config configs/diffusionvar.yaml --n_images 10000 --n_display_images 36 --save_dir dvar --classes "1_3_7" --n_class_images 1000 --ep 150
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

## Acknowledgments

- [VAR](https://github.com/FoundationVision/VAR) - Original Visual Autoregressive Modeling implementation
- [MD4](https://github.com/darioShar/pytorch-md4) - Masked Diffusion 