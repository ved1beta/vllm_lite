import torch
import time

from vllm_lite.engine.flash import flash_attention
from torch_examples.atten import pytorch_flash_attention

def test_flash_attention():
    """Test Flash Attention implementation"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return
    
    device = torch.device("cuda")
    batch_size = 2
    num_heads = 8
    seq_len = 64
    head_dim = 64
    
    print(f"🧪 Testing Flash Attention...")
    print(f"   Shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    
    # Create test tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Test without mask
    print("\n1️⃣  Testing without mask...")
    out_custom = flash_attention(Q, K, V)
    out_pytorch = pytorch_flash_attention(Q, K, V)
    
    if torch.allclose(out_custom, out_pytorch, atol=1e-4):
        print("   ✅ Results match PyTorch implementation!")
    else:
        max_diff = torch.max(torch.abs(out_custom - out_pytorch))
        print(f"   ❌ Results don't match! Max difference: {max_diff:.6f}")
    
    # Test with mask
    print("\n2️⃣  Testing with mask...")
    mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    # Create causal mask (lower triangular)
    mask = torch.tril(mask)
    
    out_custom_mask = flash_attention(Q, K, V, mask=mask)
    out_pytorch_mask = pytorch_flash_attention(Q, K, V, mask=mask)
    
    if torch.allclose(out_custom_mask, out_pytorch_mask, atol=1e-4):
        print("   ✅ Masked attention results match!")
    else:
        max_diff = torch.max(torch.abs(out_custom_mask - out_pytorch_mask))
        print(f"   ❌ Masked results don't match! Max difference: {max_diff:.6f}")
    
    print("\n🎉 Flash Attention testing completed!")

if __name__ == "__main__":
    test_flash_attention()