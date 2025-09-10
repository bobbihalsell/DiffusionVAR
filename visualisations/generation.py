import argparse
import gc
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

import dist
from utils import arg_util
from utils import misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler
from utils.image_metrics import ImageMetrics 

from train import NullDDP


def build_models_and_data(args):
    """
    Build models and data loader for evaluation.
    """
    print('[Building dataset...]')
    dataset_train, _ = build_dataset(
        args.data_path, 
        final_reso=args.data_load_reso, 
        hflip=args.hflip, 
        mid_reso=args.mid_reso
    )
    
    print('[Building data loader...]')
    ld_train = DataLoader(
        dataset=dataset_train, 
        num_workers=args.workers, 
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), 
            glb_batch_size=args.glb_batch_size, 
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, 
            fill_last=True, 
            rank=dist.get_rank(), 
            world_size=dist.get_world_size(), 
            start_ep=0, 
            start_it=0
        ),
    )
    
    print('[Building models...]')
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import build_vae_var, VQVAE
    
    vae_local, var_wo_ddp = build_vae_var(
        algo=args.algo,
        V=args.vae_vocab_size,         
        Cvae=args.vae_z_channels,          
        ch=args.vae_ch,                   
        share_quant_resi=args.vae_share_quant_resi,  
        device=dist.get_device(), 
        patch_nums=args.patch_nums, 
        cond_drop_rate=args.cond_drop_rate,
        num_classes=args.num_classes, 
        depth=args.depth, 
        shared_aln=args.saln, 
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, 
        fused_if_available=args.fuse,
        init_adaln=args.aln, 
        init_adaln_gamma=args.alng, 
        init_head=args.hd, 
        init_std=args.ini,
        drop_rate=args.drop_rate, 
        attn_drop_rate=args.attn_drop_rate,
        zero_init=args.zero_init, 
        diffusion_args=args.diffusion_args,
    )
    
    print('[Loading VAE checkpoint...]')
    vae_ckpt = getattr(args, 'vae_ckpt', 'vae_ch160v4096z32.pth')
    
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            if vae_ckpt == 'vae_ch160v4096z32.pth':
                print(f"Downloading VAE checkpoint: {vae_ckpt}")
                os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
            else:
                raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    
    dist.barrier()
    
    checkpoint = torch.load(vae_ckpt, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    vae_local.load_state_dict(state_dict, strict=True)
    print("VAE checkpoint loaded successfully")
    
    # Load VAR model checkpoint
    print('[Loading VAR model checkpoint...]')
    var_ckpt = getattr(args, 'output_dir_name', None)
    if var_ckpt and os.path.exists(var_ckpt):
        var_ckpt = os.path.join(var_ckpt, 'ckpt-best.pth')
        print(f"Loading VAR checkpoint: {var_ckpt}")
        var_checkpoint = torch.load(var_ckpt, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'trainer' in var_checkpoint:
            var_state_dict = var_checkpoint['trainer']['var_wo_ddp']
        elif 'model_state_dict' in var_checkpoint:
            var_state_dict = var_checkpoint['model_state_dict']
        else:
            var_state_dict = var_checkpoint
            
        var_wo_ddp.load_state_dict(var_state_dict, strict=True)
        print("VAR checkpoint loaded successfully")
    else:
        print("WARNING: No VAR checkpoint found. Using randomly initialized model!")
    
    vae_local = args.compile_model(vae_local, args.vfast)
    var_wo_ddp = args.compile_model(var_wo_ddp, args.tfast)

    var = (DDP if dist.initialized() else NullDDP)(
        var_wo_ddp, 
        device_ids=[dist.get_local_rank()], 
        find_unused_parameters=False, 
        broadcast_buffers=False
    )
    
    # Initialize metrics calculator
    metrics_calculator = None
    if dist.is_master():
        try:
            metrics_calculator = ImageMetrics(device=args.device)
            print("Image metrics calculator initialized")
        except Exception as e:
            print(f"Failed to initialize metrics calculator: {e}")
    
    dist.barrier()
    del dataset_train
    
    return vae_local, var_wo_ddp, var, ld_train, metrics_calculator


def build_class_dataloader(dataset, target_class, args):
    """
    Create a dataloader for a specific class.
    """
    print(f"Building dataloader for class {target_class}...")
    
    class_indices = []
    for idx in range(len(dataset)):
        try:
            _, label = dataset[idx]
            if hasattr(label, 'item'):
                label = label.item()
            if label == target_class:
                class_indices.append(idx)
        except:
            continue
    
    if not class_indices:
        print(f"No samples found for class {target_class}")
        return None
    
    print(f"Found {len(class_indices)} samples for class {target_class}")
    
    class_subset = Subset(dataset, class_indices)
    
    class_loader = DataLoader(
        dataset=class_subset, 
        num_workers=args.workers, 
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(class_subset), 
            glb_batch_size=args.glb_batch_size, 
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, 
            fill_last=True, 
            rank=dist.get_rank(), 
            world_size=dist.get_world_size(), 
            start_ep=0, 
            start_it=0
        ),
    )
    
    return class_loader


def run_overall_evaluation(args, vae_local, var_wo_ddp, ld_train, metrics_calculator):
    """
    Run overall image evaluation across all classes.
    """
    print(f"\n=== OVERALL EVALUATION ===")
    print(f"Generating {args.n_images} images for metrics calculation...")
    
    try:
        metrics = metrics_calculator.calculate_all_metrics(
            var_wo_ddp, 
            vae_local, 
            ld_train, 
            n_images=args.n_images,
            n_return_images=args.n_display_images,
            steps=args.diffusion_args.inf_steps, 
            split_batch=getattr(args, 'split_batch', 1)
        )
        
        lpips = metrics['lpips']
        inception_score = metrics['inception_score']
        inception_std = metrics['inception_std']
        fid = metrics['fid']
        generated_images = metrics['generated_images']
        real_images = metrics['real_images']
        
        print(f"Overall Results:")
        print(f"  LPIPS: {lpips:.4f}")
        print(f"  Inception Score: {inception_score:.4f} ± {inception_std:.4f}")
        print(f"  FID: {fid:.4f}")
        
        # Save images
        metrics_calculator.display_images(
            generated_images, 
            real_images, 
            dir_name=args.save_dir, 
            epoch=args.ep, 
            title=f'Overall Generated (FID: {fid:.4f})', 
            n=args.n_display_images
        )
        
        return {
            'lpips': lpips,
            'inception_score': inception_score,
            'inception_std': inception_std,
            'fid': fid
        }
        
    except Exception as e:
        print(f"Error in overall evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_class_evaluation(args, vae_local, var_wo_ddp, metrics_calculator):
    """
    Run class-conditional evaluation.
    """
    if not hasattr(args, 'classes') or args.classes is None:
        print("No classes specified for class evaluation")
        return {}
    
    class_list = [int(c) for c in args.classes.split('_')]
    print(f"\n=== CLASS-CONDITIONAL EVALUATION ===")
    print(f"Evaluating classes: {class_list}")
    print(f"Generating {args.n_class_images} images per class")
    
    # Rebuild dataset for class evaluation
    dataset_train, _ = build_dataset(
        args.data_path, 
        final_reso=args.data_load_reso, 
        hflip=args.hflip, 
        mid_reso=args.mid_reso
    )
    
    class_results = {}
    
    for class_id in class_list:
        print(f"\n--- Evaluating Class {class_id} ---")
        
        class_loader = build_class_dataloader(dataset_train, class_id, args)
        if class_loader is None:
            continue
        
        try:
            metrics = metrics_calculator.calculate_all_metrics(
                var_wo_ddp, 
                vae_local, 
                class_loader, 
                n_images=args.n_class_images,
                n_return_images=min(args.n_display_images, args.n_class_images),
                steps=args.diffusion_args.inf_steps, 
                split_batch=getattr(args, 'split_batch', 1)
            )
            
            lpips = metrics['lpips']
            inception_score = metrics['inception_score']
            inception_std = metrics['inception_std']
            fid = metrics['fid']
            generated_images = metrics['generated_images']
            real_images = metrics['real_images']
            
            print(f"Class {class_id} Results:")
            print(f"  LPIPS: {lpips:.4f}")
            print(f"  Inception Score: {inception_score:.4f} ± {inception_std:.4f}")
            print(f"  FID: {fid:.4f}")
            
            # Save class images
            class_dir = os.path.join(args.save_dir, f'class_{class_id}')
            os.makedirs(class_dir, exist_ok=True)
            
            metrics_calculator.display_images(
                generated_images, 
                real_images, 
                dir_name=class_dir, 
                epoch=args.ep, 
                title=f'Class {class_id} (FID: {fid:.4f})', 
                n=min(args.n_display_images, args.n_class_images)
            )
            
            class_results[class_id] = {
                'lpips': lpips,
                'inception_score': inception_score,
                'inception_std': inception_std,
                'fid': fid
            }
            
        except Exception as e:
            print(f"Error evaluating class {class_id}: {e}")
            import traceback
            traceback.print_exc()
        
        del class_loader
    
    del dataset_train
    return class_results


def print_summary(overall_results, class_results):
    """
    Print evaluation summary.
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if overall_results:
        print("OVERALL METRICS:")
        print(f"  LPIPS: {overall_results['lpips']:.4f}")
        print(f"  IS: {overall_results['inception_score']:.4f} ± {overall_results['inception_std']:.4f}")
        print(f"  FID: {overall_results['fid']:.4f}")
    
    if class_results:
        print("\nCLASS-CONDITIONAL METRICS:")
        for class_id, results in class_results.items():
            print(f"  Class {class_id}:")
            print(f"    LPIPS: {results['lpips']:.4f}")
            print(f"    IS: {results['inception_score']:.4f} ± {results['inception_std']:.4f}")
            print(f"    FID: {results['fid']:.4f}")
    
    print("="*60)


def main():
    """
    Main evaluation function.
    """
    # Get base args
    base_args = arg_util.init_dist_and_get_args()
    
    # Add evaluation args
    parser = argparse.ArgumentParser(description='Image evaluation for DiffusionVAR')
    parser.add_argument('--n_images', type=int, default=1000, 
                       help='Number of images for overall evaluation')
    parser.add_argument('--n_display_images', type=int, default=36,
                       help='Number of images to display/save')
    parser.add_argument('--ep', type=int, default=1,
                       help='Epoch number for naming')
    parser.add_argument('--save_dir', type=str, default='image_evaluation',
                       help='Save directory')
    parser.add_argument('--classes', type=str, default=None,
                       help='Classes for evaluation (e.g., "0_1_2_3")')
    parser.add_argument('--n_class_images', type=int, default=100,
                       help='Number of images per class')

    eval_args, _ = parser.parse_known_args()
    
    # Merge args
    for key, value in vars(eval_args).items():
        setattr(base_args, key, value)
    
    args = base_args
    args.save_dir = 'generation/' + args.save_dir
    display_dir = args.save_dir + args.exp_name
    
    print("=== EVALUATION SETTINGS ===")
    print(f"Overall images: {args.n_images}")
    print(f"Display images: {args.n_display_images}")
    print(f"Save directory: {args.save_dir}")
    if args.classes:
        print(f"Classes: {args.classes}")
        print(f"Images per class: {args.n_class_images}")
    print()
    
    # Create save directory
    if dist.is_master():
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Build everything
    print("Building components...")
    vae_local, var_wo_ddp, var, ld_train, metrics_calculator = build_models_and_data(args)
    
    if metrics_calculator is None or ld_train is None:
        print("Cannot run evaluation - missing components")
        return
    
    # Run evaluations
    overall_results = run_overall_evaluation(args, vae_local, var_wo_ddp, ld_train, metrics_calculator)
    class_results = run_class_evaluation(args, vae_local, var_wo_ddp, metrics_calculator)
    
    # Print summary
    print_summary(overall_results, class_results)
    print(f"\nResults saved to: {args.save_dir}")
    
    # Cleanup
    print("\nCleaning up...")
    del vae_local, var_wo_ddp, var, ld_train, metrics_calculator
    time.sleep(2)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    dist.barrier()
    print("Evaluation complete!")


if __name__ == '__main__':
    try:
        main()
    finally:
        dist.finalize()
        if hasattr(sys.stdout, 'close') and hasattr(sys.stderr, 'close'):
            if isinstance(sys.stdout, misc.SyncPrint):
                sys.stdout.close()
            if isinstance(sys.stderr, misc.SyncPrint):
                sys.stderr.close()