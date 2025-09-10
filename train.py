"""
Training script for VAR and DiffusionVAR models.

Modified from VAR: https://github.com/FoundationVision/VAR

This script provides the main training loop for both VAR and DiffusionVAR models
with support for distributed training, mixed precision, and logging.
"""

import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import time

import dist
from utils import arg_util
from utils import misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.image_metrics import ImageMetrics 
from utils.wandb_setup import *


def build_everything(args: arg_util.Args):
    """
    build all components for training including models, optimizers, and data loaders.
    
    this function initializes the complete training pipeline including:
    - WandB experiment tracking
    - checkpoint resuming
    - data loaders for training and validation
    - VAE and DiffusionVAR models
    - optimizer and trainer
    - image metrics calculator
    
    :param args: training configuration arguments
    :return: tuple containing all training components:
        - tb_lg: tensorboard logger
        - trainer: Trainer instance
        - start_ep: starting epoch (for resuming)
        - start_it: starting iteration (for resuming)
        - iters_train: number of training iterations per epoch
        - ld_train: training data loader iterator
        - ld_val: validation data loader
        - wandb_run: WandB run object
        - metrics_calculator: image metrics calculator
    """
    ensure_wandb_args(args)
    
    # resume training
    auto_resume_info, start_ep, start_it, trainer_state, args_state = misc.auto_resume(args, 'ckpt*.pth')
    wandb_run = init_wandb(args, auto_resume_info[0] if auto_resume_info else None)
    
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path, 
                filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'
            ), 
            verbose=True
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    metrics_calculator = None
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        dataset_train, dataset_val = build_dataset(
            args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        num_classes = args.num_classes
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size*1.5), 
            sampler=EvalDistributedSampler(
                dataset_val, 
                num_replicas=dist.get_world_size(), 
                rank=dist.get_rank()
            ),
            shuffle=False, drop_last=False,
        )
        del dataset_val
        
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), 
                glb_batch_size=args.glb_batch_size, 
                same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, 
                rank=dist.get_rank(), 
                world_size=dist.get_world_size(), 
                start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, '
              f'iters_train={iters_train}, types(tr, va)={types}')
        
        # initialize image metrics calculator if requested
        if args.calc_img_mtcs and dist.is_master():
            try:
                print("initializing image metrics calculator...")
                metrics_calculator = ImageMetrics(device=args.device)
                print("image metrics calculator initialized successfully")
            except Exception as e:
                print(f"failed to initialize image metrics calculator: {e}")
                import traceback
                traceback.print_exc()
                metrics_calculator = None
    
    else:
        num_classes = getattr(args, 'num_classes', 10)
        ld_val = ld_train = None
        iters_train = 10
    
    # build VAE and VAR imported here to reduce overhead
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import build_vae_var, VQVAE
    from trainer import Trainer
    from utils import AmpOptimizer, filter_params

    if args.algo == 'diffusion-var': name = 'DiffusionVAR'
    elif args.algo == 'var': name = 'VAR'
    else: raise ValueError(f"Invalid algorithm: {args.algo}")
    
    vae_local, var_wo_ddp = build_vae_var(
        algo=args.algo,
        V=args.vae_vocab_size,         
        Cvae=args.vae_z_channels,          
        ch=args.vae_ch,                   
        share_quant_resi=args.vae_share_quant_resi,  
        device=dist.get_device(), patch_nums=args.patch_nums, cond_drop_rate=args.cond_drop_rate,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
        diffusion_args=args.diffusion_args,
    )
    
    # load VAE checkpoint
    vae_ckpt = getattr(args, 'vae_ckpt', 'vae_ch160v4096z32.pth')
    
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            # only try to download if it's the default checkpoint
            if vae_ckpt == 'vae_ch160v4096z32.pth':
                print(f"[INFO] Downloading default VAE checkpoint: {vae_ckpt}")
                os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
            else:
                raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
        else:
            print(f"[INFO] Loading VAE checkpoint from: {vae_ckpt}")
    dist.barrier()
    # load the VAE checkpoint
    try:
        checkpoint = torch.load(vae_ckpt, map_location='cpu', weights_only=False)
        vae_local.load_state_dict(
            checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint 
            else checkpoint, 
            strict=True
        )
        print("VQVAE checkpoint loaded successfully")
        print(f"[INFO] Successfully loaded VAE from: {vae_ckpt}")
    except Exception as e:
        print(f"[ERROR] Failed to load VAE checkpoint from {vae_ckpt}: {e}")
        raise
    
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: torch.nn.Module = args.compile_model(var_wo_ddp, args.tfast)

    var: DDP = (DDP if dist.initialized() else NullDDP)(
        var_wo_ddp, 
        device_ids=[dist.get_local_rank()], 
        find_unused_parameters=False, 
        broadcast_buffers=False
    )
    
    print(f'[INIT] {name} model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([
        f'{k}={count_p(m)}' for k, m in (
            ('VAE', vae_local), 
            ('VAE.enc', vae_local.encoder), 
            ('VAE.dec', vae_local.decoder), 
            ('VAE.quant', vae_local.quantize)
        )
    ]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in ((name, var_wo_ddp),)]) + '\n\n')
    
    # build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul', 
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    trainer = Trainer(
        algo=args.algo,
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
        diffusion_args=args.diffusion_args,
    )

    if trainer_state is not None and len(trainer_state):
        print(f"loading trainer state with keys: {list(trainer_state.keys())}")
        if 'var_wo_ddp' in trainer_state:
            print(f"var_wo_ddp state keys: {list(trainer_state['var_wo_ddp'].keys())}")
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=args.pg0, prog_wp_it=20,
        )
        trainer.load_state_dict(trainer.state_dict())
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val,
        wandb_run, metrics_calculator
    )


def main_training():
    """
    main training function with comprehensive logging for DiffusionVAR models.
    
    this function orchestrates the complete training process including:
    - progressive training across multiple patch scales
    - Validation and image quality metrics evaluation
    - checkpoint saving and resuming
    - comprehensive logging to TensorBoard and WandB
    - best model tracking for both training and validation metrics
    
    step numbering strategy for WandB:
    - training step logs: Use global iteration number (g_it) - logs every 100 steps
    - epoch summary logs: Use global iteration at end of epoch (epoch * iters_per_epoch)
    - final summary: Use total iterations (total_epochs * iters_per_epoch)
    this ensures monotonically increasing step numbers for WandB.
    """
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val,
        wandb_run, metrics_calculator
    ) = build_everything(args)
    
    print(f"WandB run status: {'Initialized' if wandb_run else 'Not initialized'}")
    print(f"Metrics calculator status: {'Available' if metrics_calculator else 'Not available'}")
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    # initialize best image metrics
    best_lpips, best_fid, best_is = 999., 999., -1.
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep+1})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, 
            args, tb_lg, ld_train, iters_train, trainer, wandb_run
        )

        L_mean = stats['Lm']
        L_tail = stats['Lscale'] if 'Lscale' in stats else stats.get('Lt', -1)  # Handle both naming conventions
        acc_mean = stats['Accm'] 
        acc_tail = stats['Acct'] if 'Acct' in stats else stats.get('Accscale', -1)  # Handle both naming conventions
        grad_norm = stats['tnm']
        
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = (
            L_mean, L_tail, acc_mean, acc_tail, grad_norm
        )
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        
        # determine validation and metrics frequency
        should_eval = ((ep + 1) % args.val_freq == 0 or (ep + 1) == args.ep)
        should_calculate_image_metrics = (args.calc_img_mtcs and (ep + 1) % args.image_metrics_freq == 0)
        is_val_and_also_saving = should_eval or should_calculate_image_metrics
        
        # print epoch info for image metrics
        if should_calculate_image_metrics:
            print(f" [*] [ep{ep+1}] Calculating image metrics")
        
        if is_val_and_also_saving:
            val_loss_mean, val_loss_scales, val_acc_mean, val_acc_scales, tot, cost = trainer.eval_ep(ld_val)
            val_loss_tail = val_loss_scales[-1].item()
            val_acc_tail = val_acc_scales[-1].item()

            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = (
                min(best_val_loss_mean, val_loss_mean), 
                min(best_val_loss_tail, val_loss_tail)
            )
            best_val_acc_mean, best_val_acc_tail = (
                max(best_val_acc_mean, val_acc_mean), 
                max(best_val_acc_tail, val_acc_tail)
            )
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            
            # print comprehensive validation metrics
            print(f" [*] [ep{ep+1}] Validation - "
                  f"L_mean: {val_loss_mean:.4f} (best: {best_val_loss_mean:.4f}), "
                  f"L_tail: {val_loss_tail:.4f} (best: {best_val_loss_tail:.4f}), "
                  f"Acc_mean: {val_acc_mean:.2f} (best: {best_val_acc_mean:.2f}), "
                  f"Acc_tail: {val_acc_tail:.2f} (best: {best_val_acc_tail:.2f})")
            
            # calculate image quality metrics if available and configured
            lpips_score = 0.0
            inception_score = 0.0
            inception_std = 0.0
            fid_score = 0.0
            generated_images = None  
            
            if metrics_calculator is not None and should_calculate_image_metrics:
                try:
                    print(f"[ep{ep+1}] Calculating image quality metrics...")
                    # calculate metrics using configured number of images
                    n_images = getattr(args, 'n_images', 1000)  # Default to 1000 images
                    # get number of images for WandB logging (default to 8)
                    n_wandb_images = getattr(args, 'n_wandb_images', 8)
                    
                    metrics = metrics_calculator.calculate_all_metrics(
                        trainer.var_wo_ddp, 
                        trainer.vae_local, 
                        ld_train, 
                        n_images=n_images,
                        n_return_images=n_wandb_images,  # use n_wandb_images for WandB
                        steps=args.diffusion_args.inf_steps, 
                        split_batch=args.split_batch
                    )
                    lpips_score = metrics['lpips']
                    inception_score = metrics['inception_score']
                    inception_std = metrics['inception_std']
                    fid_score = metrics['fid']
                    generated_images = metrics['generated_images']
                    
                    # update best scores
                    best_lpips = min(best_lpips, lpips_score)
                    best_fid = min(best_fid, fid_score)
                    best_is = max(best_is, inception_score)
                    
                    # store in args for logging
                    if hasattr(args, 'lpips_score'):
                        args.lpips_score = lpips_score
                        args.inception_score = inception_score
                        args.inception_std = inception_std
                        args.fid_score = fid_score
                    
                    # add to logging dict
                    AR_ep_loss.update(
                        lpips=lpips_score, 
                        inception_score=inception_score, 
                        inception_std=inception_std, 
                        fid=fid_score
                    )
                    
                    print(f" [*] [ep{ep+1}] Image metrics - "
                          f"LPIPS: {lpips_score:.4f}, "
                          f"IS: {inception_score:.4f}±{inception_std:.4f}, "
                          f"FID: {fid_score:.4f}")
                    
                except Exception as e:
                    print(f"Error calculating image metrics: {e}")
                    import traceback
                    traceback.print_exc()
                        
            if dist.is_local_master():
                # use output_dir_name directly since it's the same as local_out_dir_path
                ckpt_dir = args.output_dir_name
                
                local_out_ckpt = os.path.join(ckpt_dir, 'ckpt-last.pth')
                local_out_ckpt_best = os.path.join(ckpt_dir, 'ckpt-best.pth')
                local_out_ckpt_pen = os.path.join(ckpt_dir, 'ckpt-pen.pth')
                
                print(f'[saving ckpt] ...', end='', flush=True)
                
                # backup system: starting from epoch 2, rename last to penultimate
                if ep >= 1:  # ep is 0-indexed, so ep >= 1 means epoch 2 or later
                    if os.path.exists(local_out_ckpt):
                        try:
                            shutil.move(local_out_ckpt, local_out_ckpt_pen)
                        except Exception as e:
                            print(f"Error moving checkpoint pen-ckpt may be incorrect: {e}")
                            import traceback
                            traceback.print_exc()
                
                # save new checkpoint as last
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                
                # update best checkpoint if validation improved
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()
        
        if is_val_and_also_saving:
            print(f'     [ep{ep+1}] Epoch Summary - Train: '
                  f'L_mean={L_mean:.4f} (best:{best_L_mean:.4f}), '
                  f'L_tail={L_tail:.4f} (best:{best_L_tail:.4f}), '
                  f'Acc_mean={acc_mean:.2f} (best:{best_acc_mean:.2f}), '
                  f'Acc_tail={acc_tail:.2f} (best:{best_acc_tail:.2f})')
            print(f'     [ep{ep+1}] Epoch Summary - Val: '
                  f'L_mean={val_loss_mean:.4f} (best:{best_val_loss_mean:.4f}), '
                  f'L_tail={val_loss_scales[-1]:.4f} (best:{best_val_loss_tail:.4f}), '
                  f'Acc_mean={val_acc_mean:.2f} (best:{best_val_acc_mean:.2f}), '
                  f'Acc_tail={val_acc_scales[-1]:.2f} (best:{best_val_acc_tail:.2f})')
        
        print(f'     [ep{ep+1}]  (training )  '
              f'Lm: {best_L_mean:.3f} ({L_mean:.3f}), '
              f'Lt: {best_L_tail:.3f} ({L_tail:.3f}),  '
              f'Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  '
              f'Remain: {remain_time},  Finish: {finish_time}', flush=True)
        
        # log to tensorboard
        tb_lg.update(head='ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        
        # log to wandb with global step for consistent comparison across batch sizes
        if wandb_run:
            current_global_step = (ep + 1) * iters_train
            
            wandb_metrics = {
                # primary x-axis: consistent across batch sizes
                'virtual_epoch': current_global_step / iters_train,
                
                # secondary x-axes for reference
                'global_step': current_global_step,
                'gradient_updates': current_global_step // args.ac,
                'epoch': ep + 1,
                
                # training metrics
                'train': {
                    'loss_mean': L_mean,
                    'loss_tail': L_tail,
                    'acc_mean': acc_mean,
                    'acc_tail': acc_tail,
                    'grad_norm': grad_norm,
                },
                'best': {
                    'loss_mean': best_L_mean,
                    'loss_tail': best_L_tail,
                    'acc_mean': best_acc_mean,
                    'acc_tail': best_acc_tail,
                },
                'lr_schedule': {
                    'learning_rate': args.cur_lr if hasattr(args, 'cur_lr') and args.cur_lr else 0,
                    'weight_decay': args.cur_wd if hasattr(args, 'cur_wd') and args.cur_wd else 0,
                }
            }
            
            if is_val_and_also_saving:
                wandb_metrics['val'] = {
                    'loss_mean': val_loss_mean,
                    'loss_tail': val_loss_tail,
                    'acc_mean': val_acc_mean,
                    'acc_tail': val_acc_tail,
                }
                wandb_metrics['best_val'] = {
                    'loss_mean': best_val_loss_mean,
                    'loss_tail': best_val_loss_tail,
                    'acc_mean': best_val_acc_mean,
                    'acc_tail': best_val_acc_tail,
                }
                # log scale losses and accuracies
                if val_loss_scales is not None and val_acc_scales is not None:
                    for i, (loss_val, acc_val) in enumerate(zip(val_loss_scales, val_acc_scales)):
                        if loss_val != -1:  # Only log valid scales
                            wandb_metrics[f'eval_scale_loss/scale_{i}'] = loss_val.item() 
                            wandb_metrics[f'eval_scale_acc/scale_{i}'] = acc_val.item()
            
            
            if should_calculate_image_metrics and metrics_calculator is not None:
                wandb_metrics['metrics'] = {
                    'lpips': lpips_score,
                    'inception_score': inception_score,
                    'inception_std': inception_std,
                    'fid': fid_score,
                }
                wandb_metrics['best_metrics'] = {
                    'lpips': best_lpips,
                    'fid': best_fid,
                    'inception_score': best_is,
                }
                log_images_to_wandb(wandb_run, current_global_step, generated_images=generated_images)
            log_to_wandb(wandb_run, wandb_metrics, current_global_step)
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   '
          f'Lm: {best_L_mean:.3f} ({L_mean}),   '
          f'Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    if wandb_run:
        # log final summary
        final_summary = {
            'final/total_time_hours': float(total_time.replace('h', '')),
            'final/best_loss_mean': best_L_mean,
            'final/best_loss_tail': best_L_tail,
            'final/best_acc_mean': best_acc_mean,
            'final/best_acc_tail': best_acc_tail,
        }
        if metrics_calculator is not None:
            final_summary.update({
                'final/best_lpips': best_lpips,
                'final/best_fid': best_fid,
                'final/best_inception_score': best_is,
            })
        
        log_to_wandb(wandb_run, final_summary, args.ep * iters_train)  # use final global iteration
        print("WandB final summary logged!")
        wandb_run.finish()
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, 
                tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, 
                trainer, wandb_run=None):
    """
    train one epoch with progressive training and comprehensive logging.
    
    This function handles a single epoch of training including:
    - progressive training stage calculation
    - learning rate and weight decay scheduling
    - training step execution with gradient accumulation
    - metric logging to console, TensorBoard, and WandB
    - time estimation for remaining training
    
    :param ep: Current epoch number
    :param is_first_ep: Whether this is the first epoch
    :param start_it: Starting iteration within epoch (for resuming)
    :param args: Training configuration arguments
    :param tb_lg: TensorBoard logger
    :param ld_or_itrt: Training data loader iterator
    :param iters_train: Total iterations per epoch
    :param trainer: Trainer instance
    :param wandb_run: WandB run object for logging
    :return: Tuple of (training_metrics_dict, time_estimation_tuple)
    """
    # import heavy packages after Dataloader object creation 
    from trainer import Trainer
    from utils.lr_control import lr_wd_annealing
    trainer: Trainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep+1:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, 
            g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        grad_norm, scale_log2, wandb_metrics = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,inp_B3HW=inp, 
            label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train, 
            wandb_run=wandb_run
        )
        grad_norm, scale_log2 = 0.0, 0.0
        if grad_norm is None:
            grad_norm = 0.0
        if scale_log2 is None:
            scale_log2 = 0.0
        
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='opt_grad/grad', grad_clip=args.tclip)
        
        if wandb_run and stepping and g_it % args.wandb_it == 0:  # log every wandb_it steps
            step_metrics = {
                'virtual_epoch': g_it / iters_train,
                
                'global_step': g_it,
                'gradient_updates': g_it // args.ac,
                'epoch': ep + 1,
                'step_in_epoch': it + 1,
                
                # training step metrics
                'train_step': {
                    'lr': max_tlr,
                    'weight_decay': max_twd,
                    'grad_norm': grad_norm,
                    'scale_log2': scale_log2,
                }
            }
            log_to_wandb(wandb_run, step_metrics, g_it, prefix="step/")

            if wandb_metrics is not None:
                wandb_run.log(wandb_metrics, step=g_it)
    
    me.synchronize_between_processes()
    return ({k: meter.global_avg for k, meter in me.meters.items()}, 
            me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15))  # +15: other cost


class NullDDP(torch.nn.Module):
    """
    Null DistributedDataParallel wrapper for single-GPU training.
    
    This class provides a no-op wrapper that mimics DDP interface but doesn't
    perform any distributed operations. Used when distributed training is disabled.
    """
    def __init__(self, module, *args, **kwargs):
        """
        Initialize NullDDP wrapper.
        
        :param module: Module to wrap
        :param args: Additional positional arguments (ignored)
        :param kwargs: Additional keyword arguments (ignored)
        """
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the wrapped module.
        
        :param args: Positional arguments for module forward
        :param kwargs: Keyword arguments for module forward
        :return: Module output
        """
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()