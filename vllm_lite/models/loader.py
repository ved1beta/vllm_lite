import os
import glob
from typing import Tuple, Any, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm_lite.utils.logger import get_logger

class DefaultModelLoader:
    """
    Default model loader with support for SafeTensors and PyTorch formats.
    Automatically detects and loads the best available model format.
    """
    
    def __init__(self, device_map: str = "auto", torch_dtype: str = "auto", 
                 trust_remote_code: bool = False):
        """
        Initialize the model loader.
        
        Args:
            device_map: Device placement strategy ('auto', 'cpu', 'cuda', etc.)
            torch_dtype: Torch data type ('auto', 'float16', 'bfloat16', 'float32')
            trust_remote_code: Whether to trust remote code in model config
        """
        self.logger = get_logger(__name__)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        
        # SafeTensors > PyTorch bin/pt
        self.format_priority = [
            ("*.safetensors", self._load_safetensors),
            ("*.bin", self._load_pytorch),
            ("*.pt", self._load_pytorch)
        ]
        
        self.logger.info(f"Initialized ModelLoader with device_map={device_map}, "
                        f"dtype={torch_dtype}, trust_remote_code={trust_remote_code}")
    
    def detect_and_load(self, model_path: str, 
                       load_config: Optional[Dict] = None) -> Tuple[Any, Any]:
        """
        Auto-detect best available format and load model + tokenizer.
        
        Args:
            model_path: Path to model directory (local) or HuggingFace model ID
            load_config: Optional dict with additional loading parameters
            
        Returns:
            Tuple of (model, tokenizer)
        """
        load_config = load_config or {}
        
        is_local = os.path.exists(model_path)
        
        if is_local:
            self.logger.info(f"Loading model from local path: {model_path}")
            for pattern, loader_func in self.format_priority:
                files = glob.glob(os.path.join(model_path, pattern))
                if files:
                    self.logger.info(f"Found {len(files)} {pattern} files, loading model...")
                    model, tokenizer = loader_func(model_path, files, load_config)
                    self._configure_tokenizer(tokenizer)
                    return model, tokenizer
            raise ValueError(f"No supported weight files found in {model_path}")
        else:
            self.logger.info(f"Loading model from HuggingFace Hub: {model_path}")
            return self._load_from_huggingface(model_path, load_config)
    
    def _configure_tokenizer(self, tokenizer):
        """Configure tokenizer with sensible defaults"""
        if tokenizer.pad_token is None:
            self.logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.padding_side is None:
            tokenizer.padding_side = "left"
    
    def _load_safetensors(self, model_path: str, files: list, 
                         load_config: Dict) -> Tuple[Any, Any]:
        """
        Load model and tokenizer from SafeTensors format.
        SafeTensors is a safe, fast, and portable format for storing tensors.
        """
        try:
            self.logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code
            )
            
            self.logger.info("Loading model from SafeTensors format...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                use_safetensors=True,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                **load_config
            )
            
            self.logger.info(f"Model loaded successfully: {model.config.model_type}")
            self.logger.info(f"Model parameters: {model.num_parameters():,}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load SafeTensors model: {e}")
            raise RuntimeError(f"Failed to load SafeTensors model: {e}")
        
    def _load_pytorch(self, model_path: str, files: list, 
                     load_config: Dict) -> Tuple[Any, Any]:
        """
        Load model and tokenizer from PyTorch format (.bin or .pt files).
        """
        try:
            self.logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code
            )
            
            self.logger.info("Loading model from PyTorch format...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype, 
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                **load_config
            )
            
            self.logger.info(f"Model loaded successfully: {model.config.model_type}")
            self.logger.info(f"Model parameters: {model.num_parameters():,}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    def _load_from_huggingface(self, model_id: str, 
                              load_config: Dict) -> Tuple[Any, Any]:
        """
        Load model directly from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')
            load_config: Additional loading parameters
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            self.logger.info(f"Downloading and loading tokenizer from HF Hub...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=self.trust_remote_code
            )
            
            self.logger.info(f"Downloading and loading model from HF Hub...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    use_safetensors=True,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=self.trust_remote_code,
                    **load_config
                )
                self.logger.info("Loaded using SafeTensors format from HF Hub")
            except Exception:
                self.logger.info("SafeTensors not available, falling back to PyTorch format...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    trust_remote_code=self.trust_remote_code,
                    **load_config
                )
                self.logger.info("Loaded using PyTorch format from HF Hub")
            
            self.logger.info(f"Model loaded successfully: {model.config.model_type}")
            self.logger.info(f"Model parameters: {model.num_parameters():,}")
            
            self._configure_tokenizer(tokenizer)
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load model from HuggingFace Hub: {e}")
            raise RuntimeError(f"Failed to load model from HuggingFace Hub: {e}")