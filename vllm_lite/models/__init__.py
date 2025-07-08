"""
Model loading and management components for vllm_lite.

This module provides:
- Model loaders for different formats (HuggingFace, GGUF, etc.)
- Quantization support (FP16, INT8, INT4)
- Memory-efficient loading with lazy loading
- Model configuration and management
"""

from .loader import DefaultModelLoader

__all__ = [
    'DefaultModelLoader',
]
