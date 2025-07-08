import os
import glob
from typing import Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DefaultModelLoader:
    def __init__(self):
        self.format_priority = [
            ("*.safetensors", self._load_safetensors),
            ("*.bin", self._load_pytorch),
            ("*.pt", self._load_pytorch)
        ]
    
    def detect_and_load(self, model_path: str) -> Tuple[Any, Any]:
        """Auto-detect best available format and load model + tokenizer"""
        for pattern, loader_func in self.format_priority:
            files = glob.glob(os.path.join(model_path, pattern))
            if files:
                print(f"Using {pattern} format")
                return loader_func(model_path, files)
        
        raise ValueError("No supported weight files found")
    
    def _load_safetensors(self, model_path: str, files: list) -> Tuple[Any, Any]:
        """Load model and tokenizer from SafeTensors format"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                use_safetensors=True,
                torch_dtype="auto",
                device_map="auto"
            )
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load SafeTensors model: {e}")
        
    def _load_pytorch(self, model_path: str, files: list) -> Tuple[Any, Any]:
        """Load model and tokenizer from PyTorch format"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto", 
                device_map="auto"
            )
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")