#!/usr/bin/env python3
"""
Setup script for vLLM-Lite: A high-throughput LLM inference engine
Inspired by vLLM but optimized for consumer hardware
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Project metadata
PACKAGE_NAME = "vllm_lite"
VERSION = "0.1.0"
DESCRIPTION = "A lightweight, high-throughput LLM inference engine for consumer hardware"
LONG_DESCRIPTION = """
vLLM-Lite is a fast and memory-efficient library for LLM inference and serving,
specifically optimized for consumer GPUs with limited VRAM. It implements core
concepts from vLLM including PagedAttention and continuous batching, while being
more accessible for personal projects and smaller deployments.

Key Features:
- PagedAttention for efficient KV cache management
- Continuous batching for high throughput
- Optimized for consumer GPUs (RTX 3050, 4060, etc.)
- Quantization support (INT8, INT4)
- CPU offloading for larger models
- HuggingFace model compatibility
"""

AUTHOR = "ved1beta"
AUTHOR_EMAIL = "ved1beta@example.com"  # Replace with your actual email
URL = "https://github.com/ved1beta/vllm-lite"  # Replace with your repo URL

# Python version requirements
PYTHON_REQUIRES = ">=3.8"

# Core dependencies
INSTALL_REQUIRES = [
    # Core ML libraries
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "tokenizers>=0.13.0",
    "accelerate>=0.20.0",
    
    # Optimization libraries
    "flash-attn>=2.0.0; platform_machine=='x86_64'",
    "xformers>=0.0.20",
    
    # Utilities
    "numpy>=1.21.0",
    "psutil>=5.9.0",
    "pynvml>=11.0.0",
    "packaging>=20.0",
    "typing-extensions>=4.0.0",
    
    # Async and serving
    "fastapi>=0.95.0",
    "uvicorn>=0.22.0",
    "pydantic>=2.0.0",
    "ray>=2.5.0",
    
    # Monitoring and logging
    "prometheus-client>=0.16.0",
    "tensorboard>=2.13.0",
    
    # Data handling
    "safetensors>=0.3.0",
    "huggingface-hub>=0.15.0",
]

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-benchmark>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "pre-commit>=3.3.0",
    ],
    "test": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-benchmark>=4.0.0",
        "coverage>=7.2.0",
    ],
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=2.0.0",
        "sphinx-autodoc-typehints>=1.23.0",
    ],
    "quantization": [
        "bitsandbytes>=0.41.0",
        "auto-gptq>=0.4.0",
    ],
    "all": [],  # Will be populated below
}

# Populate 'all' extra
EXTRAS_REQUIRE["all"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

def get_cuda_version() -> Tuple[int, int]:
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version_line = [line for line in result.stdout.split('\n') 
                       if 'release' in line][0]
        version_match = re.search(r'release (\d+)\.(\d+)', version_line)
        if version_match:
            return int(version_match.group(1)), int(version_match.group(2))
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass
    return 11, 8  # Default fallback

def get_cuda_extensions() -> List[Extension]:
    """Define CUDA extensions for custom kernels."""
    cuda_major, cuda_minor = get_cuda_version()
    
    # Common CUDA flags
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        f"-gencode=arch=compute_75,code=sm_75",  # RTX 20xx series
        f"-gencode=arch=compute_86,code=sm_86",  # RTX 30xx series
        f"-gencode=arch=compute_89,code=sm_89",  # RTX 40xx series
    ]
    
    # Architecture-specific optimizations
    if cuda_major >= 12:
        nvcc_flags.append("-gencode=arch=compute_90,code=sm_90")  # H100
    
    cxx_flags = ["-O3", "-std=c++17"]
    
    extensions = []
    
    # PagedAttention CUDA extension
    paged_attention_ext = CUDAExtension(
        name="vllm_lite._C.paged_attention",
        sources=[
            "csrc/attention/paged_attention.cu",
            "csrc/attention/attention_kernels.cu",
            "csrc/utils/cuda_utils.cu",
            "csrc/bindings/paged_attention_binding.cpp",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        include_dirs=[
            "csrc/",
            "csrc/attention/",
            "csrc/utils/",
        ],
    )
    extensions.append(paged_attention_ext)
    
    # Cache operations extension
    cache_ops_ext = CUDAExtension(
        name="vllm_lite._C.cache_ops",
        sources=[
            "csrc/cache/cache_kernels.cu",
            "csrc/cache/copy_kernels.cu",
            "csrc/bindings/cache_ops_binding.cpp",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        include_dirs=[
            "csrc/",
            "csrc/cache/",
        ],
    )
    extensions.append(cache_ops_ext)
    
    # Quantization kernels
    quant_ext = CUDAExtension(
        name="vllm_lite._C.quant_ops",
        sources=[
            "csrc/quantization/int8_kernels.cu",
            "csrc/quantization/int4_kernels.cu",
            "csrc/bindings/quant_ops_binding.cpp",
        ],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        include_dirs=[
            "csrc/",
            "csrc/quantization/",
        ],
    )
    extensions.append(quant_ext)
    
    return extensions

def check_cuda_availability():
    """Check if CUDA is available and warn if not."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Some features will be disabled.")
            return False
    except ImportError:
        print("WARNING: PyTorch not installed. Install it first.")
        return False
    return True

def read_file(filepath: str) -> str:
    """Read content from a file."""
    with open(filepath, encoding="utf-8") as f:
        return f.read()

# Check for README file
readme_content = LONG_DESCRIPTION
if os.path.exists("README.md"):
    readme_content = read_file("README.md")

# Get extensions only if CUDA is available
extensions = []
cmdclass = {}

if check_cuda_availability() and "--no-cuda" not in sys.argv:
    try:
        extensions = get_cuda_extensions()
        cmdclass["build_ext"] = BuildExtension
        print("CUDA extensions will be built.")
    except Exception as e:
        print(f"WARNING: Failed to setup CUDA extensions: {e}")
        print("Building without CUDA extensions.")

# Remove --no-cuda from sys.argv if present
if "--no-cuda" in sys.argv:
    sys.argv.remove("--no-cuda")

# Entry points for CLI tools
ENTRY_POINTS = {
    "console_scripts": [
        "vllm-lite=vllm_lite.entrypoints.cli:main",
        "vllm-lite-serve=vllm_lite.entrypoints.api_server:main",
        "vllm-lite-benchmark=vllm_lite.benchmarks.benchmark:main",
    ],
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: C++",
    "Programming Language :: CUDA",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
]

if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=readme_content,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        packages=find_packages(exclude=["tests*", "benchmarks*", "examples*"]),
        package_data={
            "vllm_lite": [
                "py.typed",
                "configs/*.yaml",
                "configs/*.json",
            ],
        },
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        ext_modules=extensions,
        cmdclass=cmdclass,
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        license="Apache License 2.0",
        keywords="llm inference attention cuda optimization machine-learning",
        project_urls={
            "Bug Reports": f"{URL}/issues",
            "Source": URL,
            "Documentation": f"{URL}#readme",
        },
        zip_safe=False,
        include_package_data=True,
    )