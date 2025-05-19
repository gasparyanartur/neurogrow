"""
Dynamic width CNN for ImageNet with PyTorch Lightning.
"""

from .model import DynamicWidthCNNLightning
from .data import CIFAR10DataModule

__all__ = [
    "DynamicWidthCNNLightning",
    "CIFAR10DataModule",
]
