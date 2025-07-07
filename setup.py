#!/usr/bin/env python3
"""
Setup script for vLLM-Lite: A lightweight, high-throughput LLM inference engine
Inspired by vLLM but optimized for consumer hardware with minimal dependencies
"""

import os
import re
import subprocess
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional

from setuptools import Extension, find_packages, setup

# Only import torch extensions if torch is available
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available during setup. CUDA extensions will be skipped.")

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
AUTHOR_EMAIL = "ved1beta@example.com"
URL = "https://github.com/ved1beta/vllm-lite"

# Python version requirements
PYTHON_REQUIRES = ">=3.8"

# Environment variables for build configuration
def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return os.getenv(name, "").lower() in ("true", "1", "yes", "on")

def get_target_device() -> str:
    """Determine target device based on environment and system."""
    target = os.getenv("VLLM_LITE_TARGET_DEVICE", "auto").lower()
    
    if target == "auto":
        if not TORCH_AVAILABLE:
            return "cpu"
        elif sys.platform.startswith("darwin"):
            return "cpu"  # macOS
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    return target

TARGET_DEVICE = get_target_device()
BUILD_CUDA = TARGET_DEVICE == "cuda" and TORCH_AVAILABLE
NO_BUILD_ISOLATION = get_env_bool("VLLM_LITE_NO_BUILD_ISOLATION")

print(f"Target device: {TARGET_DEVICE}")
print(f"Build CUDA extensions: {BUILD_CUDA}")

def get_minimal_requirements() -> List[str]:
    """Get minimal core requirements."""
    requirements = [
        "numpy>=1.21.0",
        "packaging>=20.0", 
        "typing-extensions>=4.0.0",
        "psutil>=5.9.0",
        "torch>=2.0.0",
    ]
    return requirements

def get_cuda_version() -> Tuple[int, int]:
    """Get CUDA version from nvcc."""
    if not BUILD_CUDA:
        return 11, 8
        
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
    return 11, 8

def get_cuda_extensions() -> List[Extension]:
    """Define CUDA extensions for custom kernels."""
    if not BUILD_CUDA:
        return []
        
    # Check if source directories exist
    csrc_dir = Path("csrc")
    if not csrc_dir.exists():
        print("WARNING: csrc directory not found. Skipping CUDA extensions.")
        return []
    
    cuda_major, cuda_minor = get_cuda_version()
    
    # Conservative CUDA flags for consumer GPUs
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
        # Target common consumer GPU architectures
        "-gencode=arch=compute_75,code=sm_75",  # RTX 20xx
        "-gencode=arch=compute_86,code=sm_86",  # RTX 30xx
        "-gencode=arch=compute_89,code=sm_89",  # RTX 40xx
    ]
    
    # Add newer architectures if CUDA version supports them
    if cuda_major >= 12:
        nvcc_flags.append("-gencode=arch=compute_90,code=sm_90")  # H100/Ada
    
    cxx_flags = ["-O3", "-std=c++17"]
    
    extensions = []
    
    # Define potential extensions with their source files
    extension_configs = [
        {
            "name": "vllm_lite._C.paged_attention",
            "sources": [
                "csrc/attention/paged_attention.cu",
                "csrc/attention/attention_kernels.cu",
                "csrc/utils/cuda_utils.cu",
                "csrc/bindings/paged_attention_binding.cpp",
            ],
            "include_dirs": ["csrc/", "csrc/attention/", "csrc/utils/"],
        },
        {
            "name": "vllm_lite._C.cache_ops", 
            "sources": [
                "csrc/cache/cache_kernels.cu",
                "csrc/cache/copy_kernels.cu",
                "csrc/bindings/cache_ops_binding.cpp",
            ],
            "include_dirs": ["csrc/", "csrc/cache/"],
        },
        {
            "name": "vllm_lite._C.quant_ops",
            "sources": [
                "csrc/quantization/int8_kernels.cu", 
                "csrc/quantization/int4_kernels.cu",
                "csrc/bindings/quant_ops_binding.cpp",
            ],
            "include_dirs": ["csrc/", "csrc/quantization/"],
        },
    ]
    
    # Only add extensions where all source files exist
    for config in extension_configs:
        if all(os.path.exists(src) for src in config["sources"]):
            ext = CUDAExtension(
                name=config["name"],
                sources=config["sources"],
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags,
                },
                include_dirs=config["include_dirs"],
                optional=True,  # Don't fail build if extension fails
            )
            extensions.append(ext)
            print(f"Added CUDA extension: {config['name']}")
        else:
            print(f"Skipped extension {config['name']} - missing source files")
    
    return extensions

def check_prerequisites() -> bool:
    """Check if build prerequisites are available."""
    if BUILD_CUDA:
        try:
            # Check nvcc
            subprocess.run(["nvcc", "--version"], 
                         capture_output=True, check=True)
            print("✓ NVCC found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ NVCC not found. Install CUDA toolkit.")
            return False
            
        if TORCH_AVAILABLE and not torch.cuda.is_available():
            print("✗ PyTorch CUDA not available")
            return False
            
        print("✓ CUDA prerequisites satisfied")
    
    return True

# Enhanced extras with logical groupings
EXTRAS_REQUIRE = {
    # Core ML functionality (essential for most users)
    "ml": [
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "huggingface-hub>=0.15.0",
    ],
    
    # Performance optimizations (optional but recommended)
    "perf": [
        "flash-attn>=2.0.0; platform_machine=='x86_64' and python_version>='3.8'",
        "xformers>=0.0.20; python_version>='3.8'",
    ],
    
    # Serving capabilities
    "serve": [
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.22.0",
        "pydantic>=2.0.0",
    ],
    
    # Quantization support
    "quant": [
        "bitsandbytes>=0.41.0",
        "auto-gptq>=0.4.0",
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.4.0",
    ],
    
    # Monitoring and metrics
    "monitor": [
        "pynvml>=11.0.0",
        "prometheus-client>=0.16.0",
    ],
}

# Convenience combinations
EXTRAS_REQUIRE.update({
    "standard": EXTRAS_REQUIRE["ml"] + EXTRAS_REQUIRE["serve"],
    "full": EXTRAS_REQUIRE["ml"] + EXTRAS_REQUIRE["perf"] + EXTRAS_REQUIRE["serve"],
    "all": list(set(dep for deps in EXTRAS_REQUIRE.values() for dep in deps)),
})

# Build configuration
ext_modules = []
cmdclass = {}

if BUILD_CUDA and check_prerequisites():
    try:
        extensions = get_cuda_extensions()
        if extensions:
            ext_modules = extensions
            cmdclass["build_ext"] = BuildExtension
            print(f"Will build {len(extensions)} CUDA extensions")
        else:
            print("No CUDA extensions to build")
    except Exception as e:
        print(f"WARNING: Failed to setup CUDA extensions: {e}")
        print("Building CPU-only version")

# Entry points for CLI tools
ENTRY_POINTS = {
    "console_scripts": [
        "vllm-lite=vllm_lite.entrypoints.cli:main",
        "vllm-lite-serve=vllm_lite.entrypoints.api_server:main [serve]",
        "vllm-lite-benchmark=vllm_lite.benchmarks.benchmark:main",
    ],
}

# Package data
PACKAGE_DATA = {
    "vllm_lite": [
        "py.typed",
        "configs/*.yaml",
        "configs/*.json",
    ]
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
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
]

if __name__ == "__main__":
    # Handle special flags
    if "--help-devices" in sys.argv:
        print("Supported target devices:")
        print("  cuda    - NVIDIA GPUs with CUDA")
        print("  cpu     - CPU-only execution")
        print("  auto    - Automatically detect (default)")
        print()
        print("Set with: VLLM_LITE_TARGET_DEVICE=cpu pip install -e .")
        sys.exit(0)
    
    # Remove custom flags
    custom_flags = ["--help-devices"]
    for flag in custom_flags:
        if flag in sys.argv:
            sys.argv.remove(flag)
    
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        packages=find_packages(exclude=["tests*", "benchmarks*", "examples*"]),
        package_data=PACKAGE_DATA,
        python_requires=PYTHON_REQUIRES,
        install_requires=get_minimal_requirements(),
        extras_require=EXTRAS_REQUIRE,
        ext_modules=ext_modules,
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