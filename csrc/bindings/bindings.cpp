#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of your kernel launcher template
template <typename T>
void launch_gemm_kernel_v00(
    size_t m, size_t n, size_t k,
    const T* alpha, const T* A, size_t lda,
    const T* B, size_t ldb,
    const T* beta, T* C, size_t ldc,
    cudaStream_t stream
);

template <typename T>
void launch_flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* Out,
    size_t batch_size, size_t num_heads,
    size_t seq_len_q, size_t seq_len_k, size_t head_dim,
    float scale, const T* mask,
    cudaStream_t stream
);


// Pybind11 wrapper
void gemm_launcher(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    float alpha, float beta
) {
    // Ensure tensors are contiguous and on GPU
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(C.is_cuda(), "C must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
    
    // Assume A: (m, k), B: (k, n), C: (m, n)
    size_t m = A.size(0);
    size_t k = A.size(1);
    size_t n = B.size(1);
    
    // For row-major layout: lda = k, ldb = n, ldc = n
    size_t lda = A.stride(0);  // Should be k for contiguous tensor
    size_t ldb = B.stride(0);  // Should be n for contiguous tensor  
    size_t ldc = C.stride(0);  // Should be n for contiguous tensor

    // Use default CUDA stream (0)
    cudaStream_t stream = 0;

    launch_gemm_kernel_v00<float>(
        m, n, k,
        &alpha,
        A.data_ptr<float>(), lda,
        B.data_ptr<float>(), ldb,
        &beta,
        C.data_ptr<float>(), ldc,
        stream
    );
}

// Flash Attention wrapper
void flash_attention_launcher(
    torch::Tensor Q,     // [batch_size, num_heads, seq_len_q, head_dim]
    torch::Tensor K,     // [batch_size, num_heads, seq_len_k, head_dim]
    torch::Tensor V,     // [batch_size, num_heads, seq_len_k, head_dim]
    torch::Tensor Out,   // [batch_size, num_heads, seq_len_q, head_dim]
    float scale,
    torch::Tensor mask 
) {
   
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Out.is_cuda(), "Out must be on CUDA");
    
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Out.is_contiguous(), "Out must be contiguous");
    
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D tensor");
    TORCH_CHECK(K.dim() == 4, "K must be 4D tensor");
    TORCH_CHECK(V.dim() == 4, "V must be 4D tensor");
    TORCH_CHECK(Out.dim() == 4, "Out must be 4D tensor");
    
  
    size_t batch_size = Q.size(0);
    size_t num_heads = Q.size(1);
    size_t seq_len_q = Q.size(2);
    size_t head_dim = Q.size(3);
    size_t seq_len_k = K.size(2);
    
  
    TORCH_CHECK(K.size(0) == batch_size, "K batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads, "K num_heads mismatch");
    TORCH_CHECK(K.size(3) == head_dim, "K head_dim mismatch");
    TORCH_CHECK(V.size(0) == batch_size, "V batch size mismatch");
    TORCH_CHECK(V.size(1) == num_heads, "V num_heads mismatch");
    TORCH_CHECK(V.size(2) == seq_len_k, "V seq_len mismatch");
    TORCH_CHECK(V.size(3) == head_dim, "V head_dim mismatch");
    
    const float* mask_ptr = nullptr;
    if (mask.numel() > 0) {
        TORCH_CHECK(mask.is_cuda(), "Mask must be on CUDA");
        TORCH_CHECK(mask.is_contiguous(), "Mask must be contiguous");
        mask_ptr = mask.data_ptr<float>();
    }
    
    cudaStream_t stream = 0;
    
    launch_flash_attention_kernel<float>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        Out.data_ptr<float>(),
        batch_size, num_heads,
        seq_len_q, seq_len_k, head_dim,
        scale, mask_ptr,
        stream
    );
}

// Single module declaration - exports both functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_launcher", &gemm_launcher, "GEMM kernel launcher (CUDA)");
    m.def("flash_attention_launcher", &flash_attention_launcher, "Flash Attention kernel launcher (CUDA)");
}