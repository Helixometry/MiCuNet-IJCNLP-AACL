"""MiCuNet: Multi-Curvature Learning for Emotional Manipulation Detection in Synthetic Speech"""

__version__ = "1.0.0"
__author__ = "Girish, Mohd Mujtaba Akhtar, Farhan Sheth, Muskaan Singh"
__email__ = "girish281003@gmail.com"

from .model import MiCuNet
from .layers import MixedCurvatureProjection, TemporalGating

__all__ = [
    "MiCuNet",
    "MixedCurvatureProjection",
    "TemporalGating",
]
