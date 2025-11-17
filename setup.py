from setuptools import setup, find_packages
import os

def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "A production-ready inference serving engine for Large Language Models"

def read_requirements(filename="requirements.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "safetensors>=0.4.0",
    "tokenizers>=0.15.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "numpy>=1.24.0",
    "sentencepiece>=0.1.99",
    "protobuf>=4.24.0",
    "accelerate>=0.24.0",
    "aiohttp>=3.9.0",
    "python-multipart>=0.0.6",
    "prometheus-client>=0.19.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.10.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.6.0",
        "pre-commit>=3.5.0",
        "ipython>=8.16.0",
        "ipdb>=0.13.13",
    ],
    "profiling": [
        "py-spy>=0.3.14",
        "torch-tb-profiler>=0.4.3",
        "tensorboard>=2.15.0",
        "nvidia-ml-py>=12.535.0",
    ],
    "quantization": [
        "bitsandbytes>=0.41.0",
        "auto-gptq>=0.5.0",
        "optimum>=1.14.0",
    ],
    "testing": [
        "httpx>=0.25.0",
        "locust>=2.17.0",
        "pytest-benchmark>=4.0.0",
    ],
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
    "monitoring": [
        "opentelemetry-api>=1.21.0",
        "opentelemetry-sdk>=1.21.0",
        "opentelemetry-instrumentation-fastapi>=0.42b0",
    ],
    "advanced": [
        "triton>=2.1.0",
        "flash-attn>=2.3.0",
        "xformers>=0.0.22",
    ],
}

EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name="llm-inference-engine",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready inference serving engine for Large Language Models",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-inference-engine",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "llm-server=llm_engine.server.main:main",
            "llm-benchmark=llm_engine.tools.benchmark:main",
            "llm-health-check=llm_engine.tools.health_check:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_engine": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "llm",
        "inference",
        "serving",
        "transformer",
        "language-model",
        "gpu",
        "cuda",
        "deep-learning",
        "machine-learning",
        "artificial-intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/ved1beta/vllm-lite/issues",
        "Source": "https://github.com/ved1beta/vllm-lite",
    },
)