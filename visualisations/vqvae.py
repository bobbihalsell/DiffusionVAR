import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from pathlib import Path
from PIL import Image
from torch.nn import functional as F

import matplotlib.pyplot as plt
import numpy as np

from models import VQVAE, VectorQuantizer2

vis_args = {
    'patch_nums': (1, 2, 3, 4, 5, 6, 8),
    'V': 4096,
    'Cvae': 32,
    'ch': 160,
    'share_quant_resi': 4,
    'vae_ckpt': 'vqvae/vqvae.pth',
    'n_images': 8,
    'max_res': 128,
    'data_path': 'imagenet_sample100', 
    'save_path': 'visualisations/vqvae',
}

class VAEVisualiser:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(self.args['save_path'], exist_ok=True)
        self.vae, self.ld = self.build_vqvae_batch()

    def build_vqvae_batch(self):
        """
        build vqvae and a batch of images.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load vqvae
        vae_local = VQVAE(vocab_size=self.args['V'], 
                        z_channels=self.args['Cvae'], 
                        ch=self.args['ch'], 
                        test_mode=True, 
                        share_quant_resi=self.args['share_quant_resi'], 
                        v_patch_nums=self.args['patch_nums']).to(device)

        # build simple transform: just resize and normalize to [-1, 1] (like utils/data.py)
        def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
            return x.add(x).add_(-1)
        
        simple_transform = transforms.Compose([
            transforms.Resize((self.args['max_res'], self.args['max_res'])),
            transforms.ToTensor(),
            normalize_01_into_pm1,  # same as training
        ])
        
        # load images directly from imagenet_sample folder (like utils/data.py)
        def pil_loader(path):
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            return img
        
        dataset = DatasetFolder(
            root=self.args['data_path'],  # Direct path to imagenet_sample100
            loader=pil_loader,  # same as training
            extensions=IMG_EXTENSIONS, 
            transform=simple_transform
        )
        
        # create a simple dataloader for getting 8 images
        ld = DataLoader(
            dataset,
            batch_size=self.args['n_images'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        print(f'[DataLoader] Loaded {len(dataset)}images, bs={self.args["n_images"]} from {self.args["data_path"]}')
            
        # load VAE checkpoint
        vae_ckpt = self.args['vae_ckpt']
        try:
            checkpoint = torch.load(vae_ckpt, map_location='cpu', weights_only=False)
            vae_local.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=True)
            print("VQVAE checkpoint loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load VAE checkpoint from {vae_ckpt}: {e}")
            raise
        
        return vae_local, ld
        
    def analyze_single_image(self, image):
        """
        Analyze a single image and create split visualizations
        """
        with torch.no_grad():
            image = image.to(self.device)
            
            # Get tokens from image
            ms_idx_Bl = self.vae.img_to_idxBl(image[:1])
            
            # Get direct reconstructions
            direct_recons = self.vae.idxBl_to_img(ms_idx_Bl, same_shape=True, last_one=False)
            
            # Get VAR input
            var_input = self.vae.quantize.idxBl_to_var_input(ms_idx_Bl)
            var_recons = self._reconstruct_from_var_input(var_input) if var_input is not None else []
            
            # Create split visualizations
            self.plot_tokens_only(image[0], ms_idx_Bl)
            self.plot_reconstructions_only(direct_recons, var_recons)
            
            return ms_idx_Bl, direct_recons, var_recons
    
    def plot_tokens_only(self, original_image, ms_idx_Bl):
        """
        Plot original image + colorful token maps
        Following ImageMetrics visualization pattern
        """
        num_scales = len(ms_idx_Bl)
        fig, axes = plt.subplots(1, num_scales + 1, figsize=(4 * (num_scales + 1), 4))
        
        # Convert image to display format (like ImageMetrics)
        orig_display = ((original_image + 1) / 2).clamp(0, 1)  # [-1, 1] -> [0, 1]
        orig_display = orig_display.permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]
        
        axes[0].imshow(orig_display)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Token maps for each scale
        for i, idx_Bl in enumerate(ms_idx_Bl):
            pn = self.args['patch_nums'][i]
            tokens_2d = idx_Bl[0].reshape(pn, pn).cpu().numpy()
            
            im = axes[i+1].imshow(tokens_2d, cmap='tab20', interpolation='nearest')
            axes[i+1].set_title(f'Scale {i}\n{pn}×{pn} = {pn*pn} tokens', fontsize=10)
            axes[i+1].axis('off')
        
        plt.suptitle('Original Image and Multi-Scale Token Maps', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.args['save_path'], 'tokens_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reconstructions_only(self, direct_recons, var_recons):
        """
        Plot direct reconstructions vs VAR input reconstructions
        """
        num_scales = len(direct_recons)
        fig, axes = plt.subplots(2, num_scales, figsize=(4 * num_scales, 8))
        
        # Make sure axes is 2D even for single scale
        if num_scales == 1:
            axes = axes.reshape(2, 1)
        
        # Row 1: Direct reconstructions
        for i, recon in enumerate(direct_recons):
            # Convert to display format (like ImageMetrics)
            img_display = ((recon[0] + 1) / 2).clamp(0, 1)  # [-1, 1] -> [0, 1]
            img_display = img_display.permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]
            
            axes[0, i].imshow(img_display)
            
            pn = self.args['patch_nums'][i]
            axes[0, i].set_title(f'Scale {i}\n{pn}×{pn} tokens', fontsize=16)
            axes[0, i].axis('off')
        
        # Row 2: VAR input reconstructions
        for i in range(num_scales):
            if i == 0:
                # No VAR input for scale 0 - show SOS
                axes[1, i].text(0.5, 0.5, 'SOS',
                               ha='center', va='center', transform=axes[1, i].transAxes,
                               fontsize=24, fontweight='bold', 
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
                axes[1, i].axis('off')
            elif i-1 < len(var_recons):
                # VAR reconstruction exists
                # Convert to display format (like ImageMetrics)
                img_display = ((var_recons[i-1][0] + 1) / 2).clamp(0, 1)  # [-1, 1] -> [0, 1]
                img_display = img_display.permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]
                
                axes[1, i].imshow(img_display)
                
                pn_prev = self.args['patch_nums'][i-1]
                pn_curr = self.args['patch_nums'][i]
                axes[1, i].set_title(f'Scale {i}\n{pn_prev}×{pn_prev} tokens → {pn_curr}×{pn_curr} tokens', fontsize=16)
                axes[1, i].axis('off')
            else:
                axes[1, i].axis('off')
        
        # Add row labels
        fig.text(0.02, 0.75, 'Direct Reconstruction', fontsize=18, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.25, 'VAR Model Input', fontsize=18, fontweight='bold', rotation=90, va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, hspace=0.1, wspace=0.15)  # Less left margin, more horizontal spacing
        
        plt.savefig(os.path.join(self.args['save_path'], 'reconstructions_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    

    def _reconstruct_from_var_input(self, var_input):
        """
        Convert VAR input back to image reconstructions
        """
        if var_input is None:
            return []
            
        B = var_input.shape[0]
        C = self.vae.quantize.Cvae
        reconstructions = []
        
        start_idx = 0
        for si in range(len(self.args['patch_nums']) - 1):  # VAR input has len(patch_nums)-1 entries
            scale_target = si + 1
            pn_target = self.args['patch_nums'][scale_target]
            tokens_in_scale = pn_target * pn_target
            
            # Extract features for this scale
            scale_features = var_input[:, start_idx:start_idx+tokens_in_scale, :].transpose(1, 2)
            scale_features = scale_features.view(B, C, pn_target, pn_target)
            
            # Upsample to full resolution
            H = W = self.args['patch_nums'][-1]
            if pn_target != H:
                scale_features = F.interpolate(scale_features, size=(H, W), mode='bicubic')
            
            # Decode to image
            img = self.vae.decoder(self.vae.post_quant_conv(scale_features)).clamp_(-1, 1)
            reconstructions.append(img)
            start_idx += tokens_in_scale
        return reconstructions

    def analyze_token_distribution(self):
        """
        Analyze token distribution and codebook usage across multiple images
        """
        token_stats = {
            'all_tokens': [],  # all token IDs used across all images/scales
            'tokens_per_image_per_scale': [],  # [img0: [scale0_tokens, scale1_tokens, ...], img1: [...], ...]
            'scale_names': [f'Scale {i} ({self.args["patch_nums"][i]}x{self.args["patch_nums"][i]})' for i in range(len(self.args['patch_nums']))]
        }
        
        with torch.no_grad():
            for batch_idx, (imgs, _) in enumerate(self.ld):
                # move to correct device
                imgs = imgs.to(self.device)
                # get tokens
                ms_idx_Bl = self.vae.img_to_idxBl(imgs)
                
                batch_size = imgs.shape[0]  # number of images in this batch
                
                # Process each image in the batch separately
                for img_idx in range(batch_size):
                    img_tokens_per_scale = []
                    
                    for scale_idx, idx_Bl in enumerate(ms_idx_Bl):
                        # Get tokens for this specific image at this scale
                        img_scale_tokens = idx_Bl[img_idx].flatten().cpu().numpy().tolist()
                        img_tokens_per_scale.append(img_scale_tokens)
                        token_stats['all_tokens'].extend(img_scale_tokens)
                    
                    token_stats['tokens_per_image_per_scale'].append(img_tokens_per_scale)
        
        # create distribution plots
        self.plot_token_distributions(token_stats)
        return token_stats

    def plot_token_distributions(self, token_stats):
        """
        Analyze codebook usage and token variety - per image calculations
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Distribution of used tokens (histogram)
        all_tokens = np.array(token_stats['all_tokens'])
        unique_tokens = np.unique(all_tokens)
        token_counts = np.bincount(all_tokens)
        used_token_counts = token_counts[token_counts > 0]
        
        axes[0].hist(used_token_counts, bins=30, color='lightblue', alpha=0.7)
        axes[0].set_title('Distribution of Token Usage')
        axes[0].set_xlabel('Times Used')
        axes[0].set_ylabel('Number of Tokens')
        axes[0].grid(True, alpha=0.3)
        
        # Add unused tokens and max usage stats
        total_codebook_size = 4096  # adjust if different
        unused_tokens = total_codebook_size - len(unique_tokens)
        max_usage = np.max(used_token_counts) if len(used_token_counts) > 0 else 0
        
        axes[0].text(0.7, 0.85, f'Unused tokens: {unused_tokens}\n({unused_tokens/total_codebook_size*100:.1f}%)\n\nMax usage: {max_usage}', 
                    transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        
        # Plot 2: Token diversity per scale (average per-image unique/total ratio)
        scale_diversity = []
        scale_labels = []
        
        num_scales = len(token_stats['tokens_per_image_per_scale'][0]) if token_stats['tokens_per_image_per_scale'] else 0
        
        for scale_idx in range(num_scales):
            per_image_diversity = []
            
            for img_idx, img_tokens_per_scale in enumerate(token_stats['tokens_per_image_per_scale']):
                scale_tokens = img_tokens_per_scale[scale_idx]
                if scale_tokens:
                    unique_count = len(set(scale_tokens))
                    total_count = len(scale_tokens)
                    diversity_ratio = unique_count / total_count
                    per_image_diversity.append(diversity_ratio)
            
            if per_image_diversity:
                avg_diversity = np.mean(per_image_diversity)
                scale_diversity.append(avg_diversity)
                scale_labels.append(f'S{scale_idx}')
        
        axes[1].bar(range(len(scale_diversity)), scale_diversity, color='lightgreen', alpha=0.7)
        axes[1].set_title('Avg Token Diversity per Image by Scale')
        axes[1].set_xlabel('Scale')
        axes[1].set_ylabel('Avg(Unique Tokens / Total Tokens)')
        axes[1].set_xticks(range(len(scale_labels)))
        axes[1].set_xticklabels(scale_labels)
        axes[1].grid(True, alpha=0.3)
        # axes[1].set_ylim(0, 1)
        
        # Plot 3: Token reuse across scales (average per-image overlap with previous scales)
        scale_reuse = []
        scale_labels_reuse = []
        
        for scale_idx in range(1, num_scales):  # start from scale 1
            per_image_reuse = []
            
            for img_tokens_per_scale in token_stats['tokens_per_image_per_scale']:
                current_tokens = set(img_tokens_per_scale[scale_idx])
                previous_tokens = set()
                
                # collect all tokens from previous scales for this image
                for prev_idx in range(scale_idx):
                    previous_tokens.update(img_tokens_per_scale[prev_idx])
                
                if current_tokens:
                    overlap = len(current_tokens.intersection(previous_tokens))
                    reuse_ratio = overlap / len(current_tokens)
                    per_image_reuse.append(reuse_ratio)
            
            if per_image_reuse:
                avg_reuse = np.mean(per_image_reuse)
                scale_reuse.append(avg_reuse)
                scale_labels_reuse.append(f'S{scale_idx}')
        
        if scale_reuse:
            axes[2].bar(range(len(scale_reuse)), scale_reuse, color='orange', alpha=0.7)
            axes[2].set_title('Avg Token Reuse from Previous Scales')
            axes[2].set_xlabel('Scale')
            axes[2].set_ylabel('Avg(Reused Tokens / Total Tokens)')
            axes[2].set_xticks(range(len(scale_labels_reuse)))
            axes[2].set_xticklabels(scale_labels_reuse)
            axes[2].grid(True, alpha=0.3)
            # axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args['save_path'], 'codebook_analysis.png'))
        plt.show()
        
        # Print insights
        print(f"Codebook Analysis:")
        print(f"- Using {len(unique_tokens)}/{total_codebook_size} tokens ({len(unique_tokens)/total_codebook_size*100:.1f}%)")
        print(f"- Scale diversity (avg per image): {[f'S{i}: {d:.2f}' for i, d in enumerate(scale_diversity)]}")
        if scale_reuse:
            print(f"- Scale reuse (avg per image): {[f'S{i+1}: {r:.2f}' for i, r in enumerate(scale_reuse)]}")

if __name__ == "__main__":
    # initialize visualizer
    visualizer = VAEVisualiser(vis_args)
    
    # analyze single image
    print("Analyzing single image...")
    batch = next(iter(visualizer.ld))
    img_tensor, labels = batch  # unpack the batch tuple
    img_tensor = img_tensor.to(visualizer.device)  # move to correct device
    ms_idx_Bl, direct_recons, var_recons = visualizer.analyze_single_image(img_tensor)
    
    # Analyze token distribution across all images
    print("Analyzing token distribution across dataset...")
    token_stats = visualizer.analyze_token_distribution()
    
    print(f"Analysis complete! Results saved to {vis_args['save_path']}")

