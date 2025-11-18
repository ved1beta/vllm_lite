import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open



def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_kimi(model: nn.Module, path: str):
    """
    This function is inspired by nano-vllm's approach for fast inference.
    Reference: https://github.com/GeeeekExplorer/nano-vllm
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    safetensors_files = sorted(glob(os.path.join(path, "*.safetensors")))
    
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")
    
    for file_idx, file in enumerate(safetensors_files):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                found_packed = False
                for packed_key in packed_modules_mapping:
                    if packed_key in weight_name:
                        target_param_name, shard_id = packed_modules_mapping[packed_key]
                        param_name = weight_name.replace(packed_key, target_param_name)
                        param = model.get_parameter(param_name)
                        
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        
                        loaded_weight = f.get_tensor(weight_name)
                        weight_loader(param, loaded_weight, shard_id)
                        found_packed = True
                        break
                
                if not found_packed:
                    param = model.get_parameter(weight_name)
                    
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    
                    loaded_weight = f.get_tensor(weight_name)
                    
                    weight_loader(param, loaded_weight)


#def load_model(model: nn.Module, path: str):
#    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
#    for file in glob(os.path.join(path, "*.safetensors")):
#        with safe_open(file, "pt", "cpu") as f:
#            for weight_name in f.keys():
#                for k in packed_modules_mapping:
#                    if k in weight_name:
#                        v, shard_id = packed_modules_mapping[k]https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/main
#                        param_name = weight_name.replace(k, v)
#                        param = model.get_parameter(param_name)
#                        weight_loader = getattr(param, "weight_loader")
#                        weight_loader(param, f.get_tensor(weight_name), shard_id)
#                        break
#                else:
#                    param = model.get_parameter(weight_name)
#                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
#                    weight_loader(param, f.get_tensor(weight_name))