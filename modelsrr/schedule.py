"""
Diffusion masking schedule for DiffusionVAR.

References:
- Simplified and Generalized Masked Diffusion for Discrete Data
  MD4: https://github.com/darioShar/pytorch-md4

This module implements the masking schedule used in DiffusionVAR models,
providing different masking strategies for diffusion-based training.
"""

import torch
import torch.nn as nn
import math


class MaskingSchedule(nn.Module):
    """masking noise schedule for diffusion training."""
    
    def __init__(self, schedule_fn_type='cosine', eps=1e-4):
        super().__init__()
        self.eps = eps
        
        # parse schedule type and parameters
        if 'poly' in schedule_fn_type:
            self.exponent = float(schedule_fn_type.replace('poly', ''))
            self.schedule_fn_type = 'poly'
        elif 'sigmoid' in schedule_fn_type:
            params = schedule_fn_type.replace('sigmoid', '')
            if params.startswith('_'):
                params = params[1:]  
            param_list = params.split('_') if params else []
            self.start = - float(param_list[0]) if len(param_list) > 0 and param_list[0] else -3.0
            self.end = float(param_list[1]) if len(param_list) > 1 else 3.0
            self.tau = float(param_list[2]) if len(param_list) > 2 else 1.0
            self.schedule_fn_type = 'sigmoid'
        else:
            self.schedule_fn_type = schedule_fn_type
    
    def forward(self, t):
        """return logsnr = log(alpha(t) / (1 - alpha(t)))."""
        alpha_t = self.alpha(t)
        return torch.log(alpha_t / (1.0 - alpha_t))
    
    def _dalpha(self, t):
        """derivative of _alpha with respect to t."""
        if self.schedule_fn_type == 'cosine':
            return (-math.pi / 2.0 * 
                   torch.sin(math.pi / 2.0 * (1.0 - t + self.eps)))
        elif self.schedule_fn_type == 'linear':
            return -torch.ones_like(t)
        elif 'poly' in self.schedule_fn_type:
            return -self.exponent * t ** (self.exponent - 1.0)
        elif 'sigmoid' in self.schedule_fn_type:
            # derivative of sigmoid schedule
            v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
            v_end = torch.sigmoid(torch.tensor(self.end / self.tau))
            u = (t * (self.end - self.start) + self.start) / self.tau
            sigmoid_u = torch.sigmoid(u)
            du_dt = (self.end - self.start) / self.tau
            sigmoid_derivative = sigmoid_u * (1.0 - sigmoid_u)
            return -sigmoid_derivative * du_dt / (v_end - v_start)
        else:
            raise NotImplementedError(
                f"schedule type {self.schedule_fn_type} not implemented")
    
    def dalpha(self, t):
        """scaled derivative of alpha."""
        return (1.0 - 2 * self.eps) * self._dalpha(t)
    
    def _alpha(self, t):
        """base alpha function."""
        if self.schedule_fn_type == 'linear':
            return 1.0 - t
        elif 'poly' in self.schedule_fn_type:
            return 1.0 - t**self.exponent
        elif self.schedule_fn_type == 'cosine':
            return (1.0 - 
                   torch.cos(math.pi / 2.0 * (1.0 - t + self.eps)))
        elif 'sigmoid' in self.schedule_fn_type:
            v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
            v_end = torch.sigmoid(torch.tensor(self.end / self.tau))
            u = (t * (self.end - self.start) + self.start) / self.tau
            sigmoid_u = torch.sigmoid(u)
            output = (-sigmoid_u + v_end) / (v_end - v_start)
            return torch.clamp(output, min=1e-9, max=1.0)
        else:
            raise NotImplementedError(
                f"schedule type {self.schedule_fn_type} not implemented")
    
    def alpha(self, t):
        """scaled alpha function - probability of not masking."""
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps
    
    def dgamma_times_alpha(self, t):
        """derivative of gamma times alpha."""
        return self.dalpha(t) / (1.0 - self.alpha(t))