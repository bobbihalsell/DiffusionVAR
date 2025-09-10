import time
import wandb
import os
import numpy as np
import dist
from utils import arg_util


def ensure_wandb_args(args):
    """ensure all required wandb arguments are set and print configuration."""
    print(f"WandB configuration:")
    print(f"  enable_wandb: {args.enable_wandb}")
    print(f"  wandb_project: {args.wandb_project}")
    print(f"  wandb_run_id: {args.wandb_run_id}")
    print(f"  calc_img_mtcs: {args.calc_img_mtcs}")
    print(f"  image_metrics_freq: {args.image_metrics_freq} "
          f"(every {args.image_metrics_freq} epochs)")
    print(f"  n_images: {getattr(args, 'n_images', 1000)} (for FID/metrics)")
    print(f"  n_wandb_images: {getattr(args, 'n_wandb_images', 8)} "
          f"(for WandB logging)")


def init_wandb(args: arg_util.Args, resume_info: str = None):
    """initialize wandb with comprehensive error handling and distributed training support."""
    print(f"[WandB] Starting initialization...")
    if not getattr(args, 'enable_wandb', True): 
        print("WandB is disabled via args.enable_wandb")
        return None
    
    try:
        if dist.initialized() and not dist.is_master():
            print(f"Not master process (rank {dist.get_rank()}), skipping WandB init")
            return None
    except:
        print("Distributed not available, assuming single process")
    
    # set default values
    project = getattr(args, 'wandb_project', 'diffvar') 
    run_id = getattr(args, 'wandb_run_id', None)
    
    # generate run_id if not provided
    if run_id is None:
        run_id = f"diffvar_{int(time.time())}"
        args.wandb_run_id = run_id
        print(f"Generated WandB run_id: {run_id}")
    
    # determine if we should resume
    resume_wandb = False
    if resume_info and run_id:
        print(f"Attempting to resume WandB run: {run_id}")
        resume_wandb = "auto"
    
    # prepare config
    try:
        if hasattr(args, 'state_dict'):
            config = args.state_dict()
        else:
            config = vars(args)
    except Exception as e:
        print(f"Warning: Could not get args config: {e}")
        config = {"error": "Could not serialize args"}
    
    try:
        print(f"Initializing WandB with project='{project}', run_id='{run_id}'")
        run = wandb.init(
            project=project,
            id=run_id,
            resume=resume_wandb,
            config=config,
            name=run_id,
            settings=wandb.Settings(start_method="fork")
        )

        # save the config to wandb in files accessed args.config_path
        if args.config_path and os.path.exists(args.config_path):
            try:
                run.save(args.config_path, 
                        base_path=os.path.dirname(args.config_path))
                print(f"Saved config file to WandB: {args.config_path}")
            except Exception as e:
                print(f"Warning: Could not save config file to WandB: {e}")
        else:
            print(f"Warning: Config path not found or invalid: "
                  f"{args.config_path}")
        
        print(f"WandB initialized successfully!")
        print(f"  Project: {run.project}")
        print(f"  Run ID: {run.id}")
        print(f"  Run URL: {run.url}")
        return run
    
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        import traceback
        traceback.print_exc()
        return None


def log_to_wandb(wandb_run, metrics: dict, step: int, prefix: str = ""):
    """log metrics to wandb with comprehensive error handling and data validation."""
    if wandb_run is None:
        return
    
    try:
        # flatten nested dictionaries and add prefix
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if (subvalue is not None and 
                        not (isinstance(subvalue, float) and (subvalue != subvalue))):
                        flat_metrics[f"{prefix}{key}/{subkey}"] = subvalue
            else:
                if (value is not None and 
                    not (isinstance(value, float) and (value != value))):
                    flat_metrics[f"{prefix}{key}"] = value
        
        if flat_metrics:
            wandb_run.log(flat_metrics, step=step)
            if step % 500 == 0 or len(flat_metrics) > 10:
                print(f"Logged {len(flat_metrics)} metrics to WandB at step {step}")
        else:
            print(f"No valid metrics to log at step {step}")
            
    except Exception as e:
        print(f"Error logging to WandB: {e}")
        import traceback
        traceback.print_exc()


def log_images_to_wandb(wandb_run, step: int, generated_images=None, 
                       real_images=None, max_images: int = 32):
    """log images to wandb with proper formatting and error handling."""
    if wandb_run is None:
        return
    
    try:
        def process_images(images, label_prefix):
            if images is None:
                return []
            
            images = images.detach().cpu().numpy()
            num_images = min(images.shape[0], max_images)
            images = images[:num_images]
            images = (images * 255).astype(np.uint8)
            
            wandb_images = []
            for i, img in enumerate(images):
                wandb_img = wandb.Image(img, caption=f"{label_prefix}_{i}")
                wandb_images.append(wandb_img)
            
            return wandb_images
        
        # process images
        generated_wandb_images = process_images(generated_images, "generated")
        real_wandb_images = process_images(real_images, "real")
        
        # log to wandb
        log_dict = {}
        log_dict["generated_images"] = generated_wandb_images
        log_dict["real_images"] = real_wandb_images
        
        if log_dict:
            wandb_run.log(log_dict, step=step)
            print(f"Logged {len(generated_wandb_images)} generated and "
                  f"{len(real_wandb_images)} real images")
            
    except Exception as e:
        print(f"Error logging images: {e}")