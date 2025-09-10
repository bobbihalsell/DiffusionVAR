import torch
import matplotlib.pyplot as plt
import numpy as np
import os

PATCH_NUMS = (1, 2, 3, 4)
save_dir = 'visualisations/masks'

def dvar_mask(patch_nums):
    scale_sizes = [pn * pn for pn in patch_nums]
    T = sum(scale_sizes)
    mask = torch.zeros((2*T, 2*T), dtype=torch.bool)

    # xt -> x0
    for i in range(len(scale_sizes)):
        xt_start = T + sum(scale_sizes[:i])
        xt_end = T + sum(scale_sizes[:i+1])
        x0_start = sum(scale_sizes[:i])
        x0_end = sum(scale_sizes[:i+1])
        mask[xt_start:xt_end, x0_start:x0_end] = True

    # xt -> xt
    for i in range(len(scale_sizes)):
        start = T + sum(scale_sizes[:i])
        end = T + sum(scale_sizes[:i+1])
        mask[start:end, start:end] = True

    return mask

def var_mask(patch_nums):
    scale_sizes = [pn * pn for pn in patch_nums]
    T = sum(scale_sizes)
    mask = torch.zeros((2*T, 2*T), dtype=torch.bool)
    for i, size in enumerate(scale_sizes):
        start = 2*sum(scale_sizes[:i])
        end = 2*sum(scale_sizes[:i+1])
        mask[start:end, start:end] = True
    return mask

def bd3lm_mask(patch_nums):
    scale_sizes = [pn * pn for pn in patch_nums]
    T = sum(scale_sizes)
    mask = torch.zeros((2*T, 2*T), dtype=torch.bool)

    # x0 -> x0
    for i, size in enumerate(scale_sizes):
        start = sum(scale_sizes[:i])
        end = sum(scale_sizes[:i+1])
        mask[start:end, start:end] = True

    # xt -> x0 block causal
    for i, xt_size in enumerate(scale_sizes):
        xt_start = T + sum(scale_sizes[:i])
        xt_end = T + sum(scale_sizes[:i+1])
        for j in range(i+1):
            x0_start = sum(scale_sizes[:j])
            x0_end = sum(scale_sizes[:j+1])
            mask[xt_start:xt_end, x0_start:x0_end] = True

    # xt -> xt
    for i, size in enumerate(scale_sizes):
        start = T + sum(scale_sizes[:i])
        end = T + sum(scale_sizes[:i+1])
        mask[start:end, start:end] = True

    return mask

def plot_masks_comparison(patch_nums=PATCH_NUMS, save_dir=save_dir, masks=None):
    n = len(masks)
    fig = plt.figure(figsize=(6*n, 7))  # Increased height for title space
    
    T = sum([pn*pn for pn in patch_nums])
    
    for i, (name, mask_fn) in enumerate(masks.items(), 1):
        mask = mask_fn(patch_nums)
        ax = plt.subplot(1, n, i)
        plt.imshow(mask, cmap='Blues', interpolation='none')

        # Draw gridlines
        mult = 2 if name == 'var' else 1
        block_positions = np.cumsum([pn*pn * mult for pn in patch_nums])
        block_positions = np.concatenate(([0], block_positions, [2*T]))
        for pos in block_positions:
            plt.axhline(pos - 0.5, color='black', linewidth=1)
            plt.axvline(pos - 0.5, color='black', linewidth=1)

        # Add labels based on mask type
        labels = [f"r{pn}" for pn in patch_nums]
        positions = [sum([pn*pn for pn in patch_nums[:i]]) + (pn*pn)/2 for i, pn in enumerate(patch_nums)]
        
        if name == 'var':
            # For VAR, just label the input section
            plt.xticks(positions, labels, fontsize=10)
            plt.yticks(positions, labels, fontsize=10)
            plt.xlabel("Input Tokens", fontsize=12)
            plt.ylabel("Query Tokens", fontsize=12)
        else:
            # For diffusion models, label both sections
            # Thick line for x0 / xt separation
            plt.axhline(T - 0.5, color='black', linewidth=3)
            plt.axvline(T - 0.5, color='black', linewidth=3)
            x_labels = labels + labels
            x_positions = list(positions) + [T + p for p in positions]
            plt.xticks(x_positions, x_labels, fontsize=10)
            plt.yticks(x_positions, x_labels, fontsize=10)
            plt.xlabel("Input                    Masked", fontsize=12)
            plt.ylabel("Query: Input + Masked", fontsize=12)

        plt.title(name, fontsize=14, pad=20)  # Added padding

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for titles
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'mask_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_mask(mask_fn, patch_nums=PATCH_NUMS, save_dir=save_dir, title='Diffusion Attention Mask'):
    mask = mask_fn(patch_nums)
    T = sum([pn*pn for pn in patch_nums])

    plt.figure(figsize=(10, 10))
    
    # Display mask at its natural size
    plt.imshow(mask, cmap='Blues', interpolation='none')

    # Handle gridlines based on mask size
    if mask.shape[0] == T:  # VAR case (T x T)
        # Draw gridlines for T x T
        block_positions = np.cumsum([pn*pn for pn in patch_nums])
        block_positions = np.concatenate(([0], block_positions))
        for pos in block_positions:
            plt.axhline(pos - 0.5, color='black', linewidth=1)
            plt.axvline(pos - 0.5, color='black', linewidth=1)
        
        # Labels for VAR
        labels = [f"r{pn}" for pn in patch_nums]
        positions = [sum([pn*pn for pn in patch_nums[:i]]) + (pn*pn)/2 for i, pn in enumerate(patch_nums)]
        plt.xticks(positions, labels, rotation=45, fontsize=12)
        plt.yticks(positions, labels, fontsize=12)
        plt.xlabel("Keys", fontsize=14)
        plt.ylabel("Queries", fontsize=14)
        
    else:  # 2T x 2T masks
        # Draw gridlines for 2T x 2T
        block_positions = np.cumsum([pn*pn for pn in patch_nums])
        block_positions = np.concatenate(([0], block_positions, [2*T]))
        for pos in block_positions:
            plt.axhline(pos - 0.5, color='black', linewidth=1)
            plt.axvline(pos - 0.5, color='black', linewidth=1)

        # Divide x0 and xt with a thicker line
        plt.axhline(T - 0.5, color='black', linewidth=3)
        plt.axvline(T - 0.5, color='black', linewidth=3)

        # Labels for diffusion models
        r_labels = [f"r{pn}" for pn in patch_nums]
        m_labels = [f"m{pn}" for pn in patch_nums]
        positions = [sum([pn*pn for pn in patch_nums[:i]]) + (pn*pn)/2 for i, pn in enumerate(patch_nums)]
        x_labels = r_labels + m_labels
        y_labels = r_labels + m_labels
        x_positions = list(positions) + [T + p for p in positions]
        y_positions = list(positions) + [T + p for p in positions]
        plt.xticks(x_positions, x_labels, rotation=45, fontsize=12)
        plt.yticks(y_positions, y_labels, fontsize=12)
        plt.xlabel("Keys", fontsize=14)
        plt.ylabel("Queries", fontsize=14)

    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'attention_mask.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    masks = {
        'var': var_mask,
        'diffusion-var': dvar_mask,
        'bd3lm': bd3lm_mask,
    }
    plot_mask(dvar_mask, patch_nums=PATCH_NUMS, save_dir=save_dir, title='diffusion-var')
    plot_masks_comparison(masks=masks, patch_nums=PATCH_NUMS, save_dir=save_dir)