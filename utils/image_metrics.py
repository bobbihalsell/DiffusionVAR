import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional
import os
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import dist
import time
import traceback

try:
    from lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")


class ImageMetrics:
    """calculate image quality metrics for generated images."""
    
    def __init__(self, device: torch.device):
        """initialize imagemetrics with required models and device."""
        self.device = device
        
        # initialize lpips
        if LPIPS_AVAILABLE:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, 
                                      module="torchvision")
                self.lpips_fn = LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None
        
        # initialize inception model for fid and inception score
        if FID_AVAILABLE:
            self.inception_model = InceptionV3().to(device)
            self.inception_model.eval()
        else:
            self.inception_model = None
    
    def _get_failed_metrics(self):
        """return dictionary with nan values for all metrics when generation fails."""
        return {
            'lpips': float('nan'),
            'fid': float('nan'), 
            'inception_score': float('nan'),
            'inception_std': float('nan'),
            'generated_images': torch.empty(0, 3, 256, 256),
            'real_images': torch.empty(0, 3, 256, 256)
        }
    
    def save_images_to_dir(self, images: torch.Tensor, save_dir: str, 
                          prefix: str = "img"):
        """save batch of images to directory."""
        if images is None:
            return []
            
        os.makedirs(save_dir, exist_ok=True)
        images = images.clamp(0, 1)
        
        saved_paths = []
        for i, img in enumerate(images):
            try:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                filename = f"{prefix}_{i:06d}.png"
                filepath = os.path.join(save_dir, filename)
                Image.fromarray(img_np).save(filepath)
                saved_paths.append(filepath)
            except Exception as e:
                print(f"Error saving image {i}: {e}")
        
        return saved_paths
    
    def calculate_lpips_from_dirs(self, real_dir: str, gen_dir: str) -> float:
        """calculate lpips (learned perceptual image patch similarity) from saved images."""
        if self.lpips_fn is None:
            return 0.0
        
        real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.png')])
        gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith('.png')])
        
        if len(real_files) != len(gen_files):
            print(f"Warning: Mismatch in number of images - "
                  f"Real: {len(real_files)}, Generated: {len(gen_files)}")
            min_files = min(len(real_files), len(gen_files))
            real_files = real_files[:min_files]
            gen_files = gen_files[:min_files]
        
        if len(real_files) == 0:
            return 0.0
        
        lpips_scores = []
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(real_files), batch_size):
                try:
                    batch_real_files = real_files[i:i+batch_size]
                    batch_gen_files = gen_files[i:i+batch_size]
                    
                    real_batch = []
                    gen_batch = []
                    
                    for real_f, gen_f in zip(batch_real_files, batch_gen_files):
                        # load real image
                        real_img = Image.open(os.path.join(real_dir, real_f)).convert('RGB')
                        real_tensor = torch.from_numpy(np.array(real_img)).float() / 255.0
                        real_tensor = real_tensor.permute(2, 0, 1)  # HWC -> CHW
                        real_tensor = real_tensor * 2 - 1  # [0,1] -> [-1,1] for LPIPS
                        real_batch.append(real_tensor)
                        
                        # load generated image
                        gen_img = Image.open(os.path.join(gen_dir, gen_f)).convert('RGB')
                        gen_tensor = torch.from_numpy(np.array(gen_img)).float() / 255.0
                        gen_tensor = gen_tensor.permute(2, 0, 1)  # HWC -> CHW
                        gen_tensor = gen_tensor * 2 - 1  # [0,1] -> [-1,1] for LPIPS
                        gen_batch.append(gen_tensor)
                    
                    # stack into batches and move to device
                    real_batch = torch.stack(real_batch).to(self.device)
                    gen_batch = torch.stack(gen_batch).to(self.device)
                    
                    # calculate lpips
                    lpips_batch = self.lpips_fn(real_batch, gen_batch)
                    lpips_scores.extend(lpips_batch.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in LPIPS batch {i}: {e}")
                    continue
        
        return float(np.mean(lpips_scores)) if lpips_scores else 0.0
    
    def calculate_fid_from_dirs(self, real_dir: str, gen_dir: str) -> float:
        """calculate fid (fréchet inception distance) from saved images."""
        if not FID_AVAILABLE:
            return 0.0
        
        try:
            try:
                fid = fid_score.calculate_fid_given_paths(
                    [real_dir, gen_dir], 
                    batch_size=16, 
                    device=self.device, 
                    dims=2048
                )
            except TypeError:
                print("Using older pytorch_fid API (no dims parameter)")
                fid = fid_score.calculate_fid_given_paths(
                    [real_dir, gen_dir], 
                    batch_size=16, 
                    device=self.device
                )
            
            return float(fid)
                
        except Exception as e:
            print(f"Error calculating FID: {e}")
            traceback.print_exc()
            return 0.0
    
    def calculate_inception_score_from_dir(self, gen_dir: str, 
                                         num_splits: int = 10) -> Tuple[float, float]:
        """calculate inception score from saved generated images."""
        if not FID_AVAILABLE:
            return 0.0, 0.0
        
        gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith('.png')])
        
        if len(gen_files) == 0:
            return 0.0, 0.0
        
        predictions = []
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(gen_files), batch_size):
                try:
                    batch_files = gen_files[i:i+batch_size]
                    
                    batch_images = []
                    for gen_f in batch_files:
                        gen_img = Image.open(os.path.join(gen_dir, gen_f)).convert('RGB')
                        gen_tensor = torch.from_numpy(np.array(gen_img)).float() / 255.0
                        gen_tensor = gen_tensor.permute(2, 0, 1)  # HWC -> CHW
                        batch_images.append(gen_tensor)
                    
                    batch_images = torch.stack(batch_images).to(self.device)
                    
                    # get inception predictions
                    pred = self.inception_model(batch_images)[0]  # get logits
                    pred = F.softmax(pred, dim=1)
                    predictions.append(pred.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in IS batch {i}: {e}")
                    continue
        
        if not predictions:
            return 0.0, 0.0
        
        # concatenate all predictions
        preds = np.concatenate(predictions, axis=0)
        
        # calculate inception score
        scores = []
        n_samples = preds.shape[0]
        split_size = max(1, n_samples // num_splits)
        
        for i in range(num_splits):
            if (i + 1) * split_size > n_samples:
                part = preds[i * split_size:]
            else:
                part = preds[i * split_size:(i + 1) * split_size]
            
            if len(part) == 0:
                continue
                
            kl = (part * (np.log(part + 1e-16) - 
                         np.log(np.mean(part, axis=0, keepdims=True) + 1e-16)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
        
        return float(np.mean(scores)), float(np.std(scores))

    def generate_images(self, var_model, batch_size: int = 32, 
                       label_B: torch.Tensor = None, steps: int = 250, 
                       num_classes: int = 10, split_batch: int = 1, 
                       temperature: float = 0.8) -> Optional[torch.Tensor]:
        """generate images using the var model with distributed training support."""
        world_size = dist.get_world_size() if dist.initialized() else 1
        rank = dist.get_rank() if dist.initialized() else 0
        print(f"[Rank {rank}] Starting image generation: "
              f"batch_size={batch_size}, split_batch={split_batch}")
        try:
            # generate random labels if none provided
            if label_B is None:
                label_B = torch.randint(0, num_classes, (batch_size,), 
                                      device=self.device)
            
            label_B = label_B.to(self.device)
            B = label_B.shape[0]
            
            # only master rank generates - others return dummy data
            if rank != 0:
                print(f"[Rank {rank}] Non-master rank, returning dummy data")
                return torch.zeros(batch_size, 3, 256, 256, device=self.device)
            
            # master rank does all the work
            sub_batch_size = max(1, B // max(1, split_batch))
            generated_batches = []
            
            for i in range(0, B, sub_batch_size):
                end_idx = min(i + sub_batch_size, B)
                sub_labels = label_B[i:end_idx]
                sub_B = sub_labels.shape[0]
                
                print(f"[Rank {rank}] Generating sub-batch "
                      f"{i//sub_batch_size + 1}/{split_batch}, size: {sub_B}")
                
                try:
                    # clear cuda cache before generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    sub_generated = var_model.infer_cfg(
                        B=sub_B, 
                        label_B=sub_labels, 
                        g_seed=42 + i,  
                        cfg=1.5,  
                        top_k=0,  
                        top_p=0.0,  
                        steps=steps,
                        more_smooth=False,
                    )
                    
                    if sub_generated is not None:
                        generated_batches.append(sub_generated.cpu().clone())
                        print(f"[Rank {rank}] Successfully generated sub-batch "
                              f"{i//sub_batch_size + 1}")
                    else:
                        print(f"[Rank {rank}] FAILED: Generation returned None "
                              f"for sub-batch {i//sub_batch_size + 1}")
                        return None
                        
                except Exception as e:
                    print(f"[Rank {rank}] FAILED: Error in sub-batch "
                          f"{i//sub_batch_size + 1}: {e}")
                    traceback.print_exc()
                    return None
            
            # concatenate all sub-batches
            if generated_batches:
                generated_batch = torch.cat(generated_batches, dim=0)
                print(f"[Rank {rank}] Successfully generated "
                      f"{generated_batch.shape[0]} images")
                return generated_batch
            else:
                print(f"[Rank {rank}] No successful generations")
                return None
                
        except Exception as e:
            print(f"[Rank {rank}] Critical error in image generation: {e}")
            traceback.print_exc()
            return None

    def calculate_all_metrics(self, var_model, vae_model, dataloader: DataLoader, 
                            n_images: int = 1000, n_return_images: int = 16, 
                            steps: int = 250, num_classes: int = 10, 
                            split_batch: int = 1) -> dict:
        """calculate all image quality metrics with robust error handling."""
        world_size = dist.get_world_size() if dist.initialized() else 1
        rank = dist.get_rank() if dist.initialized() else 0
        
        # non-master ranks just wait and return empty dict
        if world_size > 1 and rank != 0:
            print(f"[Rank {rank}] Non-master rank waiting for metrics calculation...")
            time.sleep(10)
            return {}

        print(f"[Rank {rank}] Starting metrics calculation for {n_images} images...")
        
        # create temporary directories
        temp_dir = tempfile.mkdtemp(prefix="image_metrics_")
        real_dir = os.path.join(temp_dir, "real")
        gen_dir = os.path.join(temp_dir, "generated")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        
        print(f"[Rank {rank}] Using temporary directory: {temp_dir}")
        
        # for return images
        generated_images = []
        real_images = []
        
        try:
            with torch.no_grad():
                total_images_processed = 0
                real_img_count = 0
                gen_img_count = 0
                
                for batch_idx, (inp_B3HW, label_B) in enumerate(dataloader):
                    if total_images_processed >= n_images:
                        break
                    
                    try:
                        # move to device
                        device = next(vae_model.parameters()).device
                        inp_B3HW = inp_B3HW.to(device, non_blocking=True)
                        label_B = label_B.to(device, non_blocking=True)
                        
                        B = inp_B3HW.shape[0]
                        remaining_images = n_images - total_images_processed
                        B_to_process = min(B, remaining_images)
                        
                        print(f"[Rank {rank}] Processing batch {batch_idx + 1}, "
                              f"{B_to_process} images")
                        
                        # generate images for this batch
                        generated_batch = self.generate_images(
                            var_model, B_to_process, label_B[:B_to_process], 
                            steps, num_classes, split_batch
                        )
                        
                        if generated_batch is None:
                            print(f"[Rank {rank}] FAILED: Generation failed "
                                  f"for batch {batch_idx + 1}")
                            print(f"[Rank {rank}] Returning NaN metrics due to "
                                  "generation failure")
                            return self._get_failed_metrics()
                        
                        # update counters
                        total_images_processed += B_to_process
                        
                        # save images to directories 
                        # [-1, 1] -> [0, 1]
                        inp_B3HW[:B_to_process] = ((inp_B3HW[:B_to_process] + 1) / 2).clamp(0, 1)
                        real_paths = self.save_images_to_dir(
                            inp_B3HW[:B_to_process], real_dir, 
                            f"real_{real_img_count:06d}"
                        )
                        # should be in [0, 1] range
                        gen_paths = self.save_images_to_dir(
                            generated_batch, gen_dir, 
                            f"gen_{gen_img_count:06d}"
                        )
                        
                        real_img_count += len(real_paths)
                        gen_img_count += len(gen_paths)
                        
                        # collect return samples HWC
                        if len(generated_images) < n_return_images:
                            n_collect = min(n_return_images - len(generated_images), 
                                          B_to_process)
                            gen_imgs = generated_batch[:n_collect].permute(0, 2, 3, 1).cpu()
                            generated_images.append(gen_imgs)
                            real_imgs = inp_B3HW[:n_collect].permute(0, 2, 3, 1).cpu()
                            real_images.append(real_imgs)
                        
                        print(f"[Rank {rank}] Batch {batch_idx + 1} complete. "
                              f"Total: {total_images_processed}/{n_images}")
                        
                        # memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"[Rank {rank}] Error in batch {batch_idx + 1}: {e}")
                        traceback.print_exc()
                        continue
            
            # combine return images
            if generated_images:
                generated_images = torch.cat(generated_images, dim=0)
                real_images = torch.cat(real_images, dim=0)
            else:
                generated_images = torch.empty(0, 3, 256, 256)
                real_images = torch.empty(0, 3, 256, 256)
            
            # check if we have any generated images
            if gen_img_count == 0:
                print(f"[Rank {rank}] FAILED: No generated images were saved, "
                      "returning NaN metrics")
                return self._get_failed_metrics()
            
            # calculate metrics
            print(f"[Rank {rank}] Calculating metrics from saved images...")
            metrics = {}
            
            # LPIPS
            try:
                if LPIPS_AVAILABLE:
                    print(f"[Rank {rank}] Calculating LPIPS...")
                    metrics['lpips'] = self.calculate_lpips_from_dirs(real_dir, gen_dir)
                    print(f"[Rank {rank}] LPIPS: {metrics['lpips']:.4f}")
                else:
                    metrics['lpips'] = 0.0
            except Exception as e:
                print(f"[Rank {rank}] FAILED: LPIPS calculation failed: {e}")
                metrics['lpips'] = float('nan')
            
            # FID
            try:
                if FID_AVAILABLE:
                    print(f"[Rank {rank}] Calculating FID...")
                    metrics['fid'] = self.calculate_fid_from_dirs(real_dir, gen_dir)
                    print(f"[Rank {rank}] FID: {metrics['fid']:.4f}")
                else:
                    metrics['fid'] = 0.0
            except Exception as e:
                print(f"[Rank {rank}] FAILED: FID calculation failed: {e}")
                metrics['fid'] = float('nan')
            
            # inception score
            try:
                if FID_AVAILABLE:
                    print(f"[Rank {rank}] Calculating inception score...")
                    inception_score, inception_std = self.calculate_inception_score_from_dir(gen_dir)
                    metrics['inception_score'] = inception_score
                    metrics['inception_std'] = inception_std
                    print(f"[Rank {rank}] inception score: "
                          f"{inception_score:.4f} ± {inception_std:.4f}")
                else:
                    metrics['inception_score'] = 0.0
                    metrics['inception_std'] = 0.0
            except Exception as e:
                print(f"[Rank {rank}] FAILED: inception score calculation failed: {e}")
                metrics['inception_score'] = float('nan')
                metrics['inception_std'] = float('nan')
            
            # return images
            metrics['generated_images'] = generated_images
            metrics['real_images'] = real_images
            
            print(f"[Rank {rank}] Metrics calculation complete!")
            
        except Exception as e:
            print(f"[Rank {rank}] FAILED: Critical error in metrics calculation: {e}")
            traceback.print_exc()
            metrics = self._get_failed_metrics()
        
        finally:
            # cleanup
            print(f"[Rank {rank}] Cleaning up temporary directory: {temp_dir}")
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Rank {rank}] Cleanup failed: {e}")
        
        return metrics

    def display_images(self, gen_images: torch.Tensor, real_images: torch.Tensor = None, 
                    dir_name: str = 'image_evaluation', epoch: int = 1, 
                    title: str = "Generated Images", name: str = "generated_images", 
                    n: int = 8, grid_cols: int = None):
        """Display images in clean grid layouts."""
        if len(gen_images) == 0:
            print("No images to display")
            return

        os.makedirs(dir_name, exist_ok=True)

        # convert to numpy and sample
        gen_images_np = gen_images.detach().cpu().numpy()   
        if n < len(gen_images_np):
            idx = np.random.choice(len(gen_images_np), size=n, replace=False)
            gen_images_np = gen_images_np[idx]
        
        real_images_np = None
        if real_images is not None:
            real_images_np = real_images.detach().cpu().numpy()
            if n < len(real_images_np):
                idx = np.random.choice(len(real_images_np), size=n, replace=False)
                real_images_np = real_images_np[idx]
        
        # calculate grid dimensions
        if grid_cols is None:
            grid_cols = min(8, int(np.ceil(np.sqrt(len(gen_images_np)))))
        grid_rows = int(np.ceil(len(gen_images_np) / grid_cols))
        
        # generated images only
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.2, grid_rows * 1.2))
        fig.suptitle(f"{title} - Epoch {epoch}", fontsize=10, y=0.95)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.05, hspace=0.1)
        
        if len(gen_images_np) == 1:
            axes = [axes]
        elif grid_rows == 1:
            axes = [axes] if grid_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(grid_rows * grid_cols):
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            if i < len(gen_images_np):
                ax.imshow(gen_images_np[i])
                ax.set_title(f"Gen {i+1}", fontsize=8, pad=2)
            ax.axis('off')
        
        gen_save_path = os.path.join(dir_name, f"{name}_generated_epoch_{epoch}.png")
        plt.savefig(gen_save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Saved generated images to: {gen_save_path}")
        
        # side-by-side comparison if real images provided
        if real_images_np is not None:
            fig, axes = plt.subplots(grid_rows, 2 * grid_cols, figsize=(grid_cols * 2.4, grid_rows * 1.2))
            fig.suptitle(f"Real vs {title} - Epoch {epoch}", fontsize=10, y=0.95)
            plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.02, hspace=0.1)
            
            if grid_rows == 1 and 2 * grid_cols == 1:
                axes = np.array([[axes]])
            elif grid_rows == 1:
                axes = axes.reshape(1, -1)
            elif 2 * grid_cols == 1:
                axes = axes.reshape(-1, 1)
            
            left_half = axes[:, :grid_cols] if grid_rows > 1 else axes[:grid_cols]
            right_half = axes[:, grid_cols:] if grid_rows > 1 else axes[grid_cols:]
            
            if grid_rows == 1:
                left_axes_flat = left_half if isinstance(left_half, np.ndarray) else [left_half]
                right_axes_flat = right_half if isinstance(right_half, np.ndarray) else [right_half]
            else:
                left_axes_flat = left_half.flatten()
                right_axes_flat = right_half.flatten()
            
            # plot real images
            for i in range(grid_rows * grid_cols):
                if i < len(left_axes_flat):
                    ax = left_axes_flat[i]
                    if i < len(real_images_np):
                        ax.imshow(real_images_np[i])
                        ax.set_title(f"Real {i+1}", fontsize=8, pad=2)
                    ax.axis('off')
            
            # plot generated images
            for i in range(grid_rows * grid_cols):
                if i < len(right_axes_flat):
                    ax = right_axes_flat[i]
                    if i < len(gen_images_np):
                        ax.imshow(gen_images_np[i])
                        ax.set_title(f"Gen {i+1}", fontsize=8, pad=2)
                    ax.axis('off')
            
            fig.text(0.25, 0.02, 'REAL', fontsize=9, ha='center', weight='bold', alpha=0.7)
            fig.text(0.75, 0.02, 'GENERATED', fontsize=9, ha='center', weight='bold', alpha=0.7)
            
            comp_save_path = os.path.join(dir_name, f"{name}_comparison_epoch_{epoch}.png")
            plt.savefig(comp_save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
            print(f"Saved comparison images to: {comp_save_path}")
            
            return gen_save_path, comp_save_path
        
        return gen_save_path