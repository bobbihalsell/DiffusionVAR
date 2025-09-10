import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union
import yaml

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      '
          f'pip3 install typed-argument-parser     <<<<<<<<', 
          file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      '
          f'pip3 install typed-argument-parser     <<<<<<<<', 
          file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import dist


class DictObj:
    """object for dot notation access to dicts."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class Args(Tap):
    """args class that extends Tap with YAML loading capabilities."""
    
    # runtime attributes that are automatically set during training
    cmd: str = ' '.join(sys.argv[1:])
    branch: str = '[unknown]'
    commit_id: str = '[unknown]'
    commit_msg: str = '[unknown]'
    
    # try to get git info, but don't fail if not in a git repo
    try:
        branch = subprocess.check_output(
            f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', 
            shell=True
        ).decode('utf-8').strip() or '[unknown]'
        commit_id = subprocess.check_output(
            f'git rev-parse HEAD', shell=True
        ).decode('utf-8').strip() or '[unknown]'
        commit_msg = (subprocess.check_output(
            f'git log -1', shell=True
        ).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()
    except:
        pass  # Use default values if git commands fail
    acc_mean: float = None
    acc_tail: float = None
    L_mean: float = None
    L_tail: float = None
    vacc_mean: float = None
    vacc_tail: float = None
    vL_mean: float = None
    vL_tail: float = None
    grad_norm: float = None
    cur_lr: float = None
    cur_wd: float = None
    cur_it: str = ''
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    lpips_score: float = None
    inception_score: float = None
    inception_std: float = None
    fid_score: float = None
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Create Args instance from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # create instance
        args = cls()
        
        # set all values from YAML
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(args, key, DictObj(value))
            else:
                setattr(args, key, value)
        
        return args
    
    def seed_everything(self, benchmark: bool):
        """Set up deterministic training."""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        
        seed = getattr(self, 'seed', None)
        if seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:
        """Get random generator for distributed training."""
        seed = getattr(self, 'seed', None)
        if seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(seed * dist.get_world_size() + dist.get_rank())
        return g
    
    def compile_model(self, model, fast_mode: int):
        """Compile model with torch.compile if requested."""
        local_debug = getattr(self, 'local_debug', False)
        if fast_mode == 0 or local_debug:
            return model
        
        if not hasattr(torch, 'compile'):
            return model
            
        mode_map = {
            1: 'reduce-overhead',
            2: 'max-autotune', 
            3: 'default',
        }
        return torch.compile(model, mode=mode_map.get(fast_mode, 'default'))
    
    @staticmethod
    def set_tf32(tf32: bool):
        """Configure TensorFloat32 settings."""
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device'}:
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep:
            with open(self.log_txt_path, 'w') as fp:
                json.dump({
                    'is_master': dist.is_master(), 
                    'name': self.exp_name, 
                    'cmd': self.cmd, 
                    'commit': self.commit_id, 
                    'branch': self.branch, 
                    'tb_log_dir_path': self.tb_log_dir_path
                }, fp, indent=0)
                fp.write('\n')
        
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time, 
            'lpips_score': self.lpips_score, 
            'inception_score': self.inception_score, 
            'inception_std': self.inception_std, 
            'fid_score': self.fid_score
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')

    def __str__(self):
        """String representation using actual instance attributes."""
        s = []
        for k, v in sorted(vars(self).items()):
            if not k.startswith('_') and k not in {'device'}:
                s.append(f'  {k:20s}: {v}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}'
    
    def set_runtime_attributes(self):
        """Set runtime attributes that are computed during initialization."""
        # git info
        try:
            self.branch = subprocess.check_output(
                f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', 
                shell=True
            ).decode('utf-8').strip() or '[unknown]'
            self.commit_id = subprocess.check_output(
                f'git rev-parse HEAD', shell=True
            ).decode('utf-8').strip() or '[unknown]'
            commit_lines = subprocess.check_output(
                f'git log -1', shell=True
            ).decode('utf-8').strip().splitlines() or ['[unknown]']
            self.commit_msg = commit_lines[-1].strip()
        except:
            self.branch = '[unknown]'
            self.commit_id = '[unknown]'
            self.commit_msg = '[unknown]'
        
        # command line
        self.cmd = ' '.join(sys.argv[1:])


def init_dist_and_get_args(config_path: str = None, default_config_path: str = 'configs/default.yaml'):
    """Initialize distributed training and get arguments."""
    # clean up distributed args
    cleaned_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--local-rank=') or arg.startswith('--local_rank='):
            i += 1
            continue
        elif arg in ['--local-rank', '--local_rank'] and i + 1 < len(sys.argv):
            i += 2
            continue
        else:
            cleaned_argv.append(arg)
            i += 1
    
    sys.argv = cleaned_argv
    
    # look for --config in command line args if config_path not provided
    if config_path is None:
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                del sys.argv[i:i+2]
                break
            elif arg.startswith('--config='):
                config_path = arg.split('=', 1)[1]
                del sys.argv[i]
                break
    
    # load config
    if config_path and os.path.exists(config_path):
        print(f"Loading config: {config_path}")
        args = Args.from_yaml(config_path)
    elif default_config_path and os.path.exists(default_config_path):
        print(f"Loading default config: {default_config_path}")
        args = Args.from_yaml(default_config_path)
    else:
        raise ValueError(f"Config file not found: {config_path or default_config_path}")

    args.config_path = config_path
    
    # override with remaining command line args using Tap's built-in parsing
    if len(sys.argv) > 1:
        cmd_args = Args(explicit_bool=True).parse_args(known_only=True)
        for key, value in vars(cmd_args).items():
            if hasattr(args, key) and getattr(args, key) != value:
                setattr(args, key, value)
                print(f"Override: {key} = {value}")
    
    # set runtime attributes
    args.set_runtime_attributes()
    
    # apply local debug overrides if needed
    if getattr(args, 'local_debug', False) or 'KEVIN_LOCAL' in os.environ:
        print("Applying local debug overrides...")
        args.pn = '1_2_3'
        args.seed = 1
        args.aln = 1e-2
        args.alng = 1e-5
        args.saln = False
        args.afuse = False
        args.pg = 0.8
        args.pg0 = 1
    
    # validate required paths
    if not args.data_path or args.data_path == '/path/to/imagenet':
        raise ValueError("Data path not specified! Please set 'data_path' in your config file")
    
    # initialize distributed training
    from utils import misc
    local_out_dir_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        args.output_dir_name
    )
    args.local_out_dir_path = local_out_dir_path
    os.makedirs(local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=local_out_dir_path, timeout=30)
    
    # set environment
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=args.pg == 0)
    args.device = dist.get_device()
    
    # process patch numbers (calculated values)
    pn = args.pn
    if pn == '256':
        pn = '1_2_3_4_5_6_8_10_13_16'
    elif pn == '512':
        pn = '1_2_3_4_6_9_13_18_24_32'
    elif pn == '1024':
        pn = '1_2_3_4_5_7_9_12_16_21_27_36_48_64'
    args.pn = pn
    
    args.patch_nums = tuple(map(int, pn.replace('-', '_').split('_')))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    
    # process batch size and learning rate (calculated values)
    bs_per_gpu = round(args.bs / args.ac / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = bs_per_gpu * dist.get_world_size()
    args.workers = min(max(0, args.workers), args.batch_size)
    args.split_batch = args.batch_size // args.image_bs
    
    # learning rate scaling (calculated)
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    args.twde = args.twde or args.twd  # Use twde if set, otherwise twd
    
    # warmup settings (calculated)
    if args.wp == 0:
        args.wp = args.ep * 1/50
    
    # progressive training (calculated)
    if args.pgwp == 0:
        args.pgwp = args.ep * 1/300
    
    if args.pg > 0:
        args.sche = f'lin{args.pg:g}'
    
    # set up paths (calculated)
    args.log_txt_path = os.path.join(local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(local_out_dir_path, f'ar-ckpt-last.pth')
    
    # tensorboard path (calculated)
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-VARd{args.depth}'
        f'__pn{pn}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.tblr:g}wd{args.twd:g}'
    )
    args.tb_log_dir_path = os.path.join(local_out_dir_path, tb_name)
    
    return args