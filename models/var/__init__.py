"""
VAR model components.

Modified from VAR: https://github.com/FoundationVision/VAR

This module provides the VAR model implementation with transformer backbone
and basic building blocks for autoregressive image generation.
"""

from models.dvar.var import VAR

__all__ = ['VAR']