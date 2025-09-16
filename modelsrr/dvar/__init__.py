"""
DiffusionVAR model components.

Modified from VAR: https://github.com/FoundationVision/VAR

This module provides the DiffusionVAR model implementation with diffusion scheduling
and transformer backbone for conditional image generation.
"""

from .dvar import DiffusionVAR
from .var import VAR

__all__ = ['DiffusionVAR', 'VAR']