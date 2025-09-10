#!/usr/bin/env python3
"""
Script to measure generation time per batch for DiffusionVAR.
"""

import argparse
import time
import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

import dist
from utils import arg_util
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler
from torch.utils.data import DataLoader

def time_generation(args, num_batches=10):
    """
    Time generation for a specified number of batches.
    """
    print(f"Timing generation for {num_batches} batches...")
    print(f"Batch size: {args.glb_batch_size}")
    print(f"Diffusion steps: {args.diffusion_args.inf_steps}")
    print()
    
    # Build dataset and data loader
    print("Building dataset...")
    dataset_train, _ = build_dataset(
        args.data_path, 
        final_reso=args.data_load_reso, 
        hflip=args.hflip, 
        mid_reso=args.mid_reso
    )
    
    print("Building data loader...")
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
    
    # Build models
    print("Building models...")
    from train import build_models
    vae_local, var_wo_ddp, var = build_models(args)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    var_wo_ddp = var_wo_ddp.to(device)
    vae_local = vae_local.to(device)
    
    # Set to eval mode
    var_wo_ddp.eval()
    vae_local.eval()
    
    print("Starting timing...")
    print()
    
    # Warmup
    print("Warming up (2 batches)...")
    with torch.no_grad():
        for i, batch in enumerate(ld_train):
            if i >= 2:
                break
            _ = var_wo_ddp.sample(
                batch_size=args.glb_batch_size,
                steps=args.diffusion_args.inf_steps,
                vae_local=vae_local
            )
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Time generation
    print(f"Timing {num_batches} batches...")
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(ld_train):
            if i >= num_batches:
                break
                
            start_time = time.time()
            
            # Generate images
            generated = var_wo_ddp.sample(
                batch_size=args.glb_batch_size,
                steps=args.diffusion_args.inf_steps,
                vae_local=vae_local
            )
            
            end_time = time.time()
            batch_time = end_time - start_time
            times.append(batch_time)
            
            print(f"Batch {i+1:2d}: {batch_time:.3f}s")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print()
    print("=== TIMING RESULTS ===")
    print(f"Batches timed: {len(times)}")
    print(f"Average time per batch: {avg_time:.3f}s")
    print(f"Min time per batch: {min_time:.3f}s")
    print(f"Max time per batch: {max_time:.3f}s")
    print(f"Images per second: {args.glb_batch_size / avg_time:.2f}")
    print(f"Diffusion steps: {args.diffusion_args.inf_steps}")
    print(f"Batch size: {args.glb_batch_size}")
    
    return avg_time

def main():
    """Main function."""
    # Get base args
    base_args = arg_util.init_dist_and_get_args()
    
    # Add timing args
    parser = argparse.ArgumentParser(description='Time DiffusionVAR generation')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of batches to time (default: 10)')
    
    timing_args, _ = parser.parse_known_args()
    
    # Merge args
    for key, value in vars(timing_args).items():
        setattr(base_args, key, value)
    
    args = base_args
    
    print("=== GENERATION TIMING ===")
    print(f"Config: {args.config}")
    print(f"Number of batches: {args.num_batches}")
    print()
    
    # Time generation
    avg_time = time_generation(args, args.num_batches)
    
    print(f"\nAverage generation time per batch: {avg_time:.3f} seconds")

if __name__ == '__main__':
    try:
        main()
    finally:
        dist.finalize()
