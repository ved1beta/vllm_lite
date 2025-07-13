#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of Flash Attention kernel launcher template
template <typename T>
void launch_flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* Out,
    size_t batch_size, size_t num_heads,
    size_t seq_len_q, size_t seq_len_k, size_t head_dim,
    float scale, const T* mask,
    cudaStream_t stream
);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_launcher", &flash_attention_launcher, "Flash Attention kernel launcher (CUDA)");
}
