import torch
from torch.utils.cpp_extension import load
import os

# Get the absolute path to the csrc directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up to project root
csrc_dir = os.path.join(project_root, "csrc")

# Load the extension with correct file paths
matrix_ext = load(
    name="matrix_ext",
    sources=[
        os.path.join(csrc_dir, "matrix.cu"),        
        os.path.join(csrc_dir, "bindings", "bindings.cpp")       
    ],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-arch=sm_86"]
)

def gemm(A, B, alpha=1.0, beta=0.0):
    """
    Compute C = alpha * A @ B + beta * C
    
    Args:
        A: Tensor of shape (m, k)
        B: Tensor of shape (k, n)
        alpha: Scalar multiplier for A @ B
        beta: Scalar multiplier for C (initial values)
    
    Returns:
        C: Tensor of shape (m, n)
    """
    # Ensure inputs are on the same device and have correct dtype
    assert A.device == B.device, "A and B must be on the same device"
    assert A.dtype == B.dtype, "A and B must have the same dtype"
    assert A.dtype == torch.float32, "Only float32 is supported"
    
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"Matrix dimension mismatch: A.shape[1]={k} != B.shape[0]={k2}"
    
    # Create output tensor
    C = torch.zeros((m, n), device=A.device, dtype=A.dtype)
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    # Call the CUDA kernel
    matrix_ext.gemm_launcher(A, B, C, float(alpha), float(beta))
    
    return C

# Test function
def test_gemm():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        A = torch.randn(64, 32, device=device, dtype=torch.float32)
        B = torch.randn(32, 48, device=device, dtype=torch.float32)
        
        # Test our implementation
        C_custom = gemm(A, B, alpha=1.0, beta=0.0)
        
        # Test against PyTorch's implementation
        C_torch = torch.matmul(A, B)
        
        # Check if results are close
        if torch.allclose(C_custom, C_torch, atol=1e-5):
            print("✅ Test passed! Results match PyTorch implementation")
        else:
            print("❌ Test failed! Results don't match")
            print(f"Max difference: {torch.max(torch.abs(C_custom - C_torch))}")
    else:
        print("CUDA not available, skipping test")

if __name__ == "__main__":
    test_gemm()