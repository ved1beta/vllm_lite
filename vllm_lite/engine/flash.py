import torch
import math
from torch_examples.atten import pytorch_flash_attention
from torch.utils.cpp_extension import load
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
csrc_dir = os.path.join(project_root, "csrc")

try:
    attention_ext = load(
        name="attention_ext",
        sources=[
            os.path.join(csrc_dir, "flash_atten.cu"),
            os.path.join(csrc_dir, "bindings", "attention_bindings.cpp")
        ],
        verbose=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_86"]
    )
    CUDA_AVAILABLE = True
    print("✅ Flash Attention CUDA extension loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load Flash Attention CUDA extension: {e}")
    print("💡 Falling back to PyTorch implementation")
    CUDA_AVAILABLE = False

def flash_attention(Q, K, V, mask=None, scale=None):
    """
    Flash Attention implementation
    
    Args:
        Q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        K: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
        V: Value tensor [batch_size, num_heads, seq_len_k, head_dim]
        mask: Optional mask [batch_size, seq_len_q, seq_len_k]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch_size, num_heads, seq_len_q, head_dim]
    """
    batch_size, num_heads, seq_len_q, head_dim = Q.shape
    seq_len_k = K.shape[2]
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    assert Q.shape == (batch_size, num_heads, seq_len_q, head_dim)
    assert K.shape == (batch_size, num_heads, seq_len_k, head_dim)
    assert V.shape == (batch_size, num_heads, seq_len_k, head_dim)
    
    if mask is not None:
        assert mask.shape == (batch_size, seq_len_q, seq_len_k)

    Out = torch.zeros_like(Q)
    
    if CUDA_AVAILABLE and Q.is_cuda:
        try:
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            Out = Out.contiguous()
            
            if mask is not None:
                mask = mask.contiguous().float()
            else:
                mask = torch.empty(0, device=Q.device, dtype=Q.dtype)
            
            attention_ext.flash_attention_launcher(Q, K, V, Out, float(scale), mask)
            return Out
            
        except Exception as e:
            print(f" CUDA Flash Attention failed: {e}")
            print("Falling back to PyTorch implementation")
    
    # Fallback to PyTorch implementation
    return pytorch_flash_attention(Q, K, V, mask, scale)

