#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_LAST_CUDA_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
        } \
    } while (0)

template <typename T>
__global__ void flash_attention_kernel(
    const T* Q,        // [batch_size, num_heads, seq_len_q, head_dim]
    const T* K,        // [batch_size, num_heads, seq_len_k, head_dim]
    const T* V,        // [batch_size, num_heads, seq_len_k, head_dim]
    T* Out,            // [batch_size, num_heads, seq_len_q, head_dim]
    size_t batch_size,
    size_t num_heads,
    size_t seq_len_q,
    size_t seq_len_k,
    size_t head_dim,
    float scale,
    const T* mask
) {
    // Thread indices
    size_t q_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    size_t d_idx = blockIdx.y * blockDim.y + threadIdx.y;  
    size_t batch_head_idx = blockIdx.z;  
    
    if (q_idx >= seq_len_q || d_idx >= head_dim || batch_head_idx >= batch_size * num_heads) {
        return;
    }
  
    size_t batch_idx = batch_head_idx / num_heads;
    size_t head_idx = batch_head_idx % num_heads;
    
    // Calculate base offsets
    size_t base_offset = batch_idx * num_heads * seq_len_q * head_dim + head_idx * seq_len_q * head_dim;
    size_t k_base_offset = batch_idx * num_heads * seq_len_k * head_dim + head_idx * seq_len_k * head_dim;
    size_t v_base_offset = batch_idx * num_heads * seq_len_k * head_dim + head_idx * seq_len_k * head_dim;
    

    T max_score = -1e9f;
    T sum_exp = 0.0f;
    T output_val = 0.0f;
 
    for (size_t k_pos = 0; k_pos < seq_len_k; ++k_pos) {
        T score = 0.0f;
        
        // Compute dot product Q[q_idx] · K[k_pos]
        for (size_t d = 0; d < head_dim; ++d) {
            size_t q_elem_idx = base_offset + q_idx * head_dim + d;
            size_t k_elem_idx = k_base_offset + k_pos * head_dim + d;
            score += Q[q_elem_idx] * K[k_elem_idx];
        }
        
        score *= scale;
        
        if (mask != nullptr) {
            size_t mask_idx = batch_idx * seq_len_q * seq_len_k + q_idx * seq_len_k + k_pos;
            if (mask[mask_idx] == 0) {
                score = -1e9f;
            }
        }
        
        max_score = fmaxf(max_score, score);
    }
    

    for (size_t k_pos = 0; k_pos < seq_len_k; ++k_pos) {
        T score = 0.0f;

        for (size_t d = 0; d < head_dim; ++d) {
            size_t q_elem_idx = base_offset + q_idx * head_dim + d;
            size_t k_elem_idx = k_base_offset + k_pos * head_dim + d;
            score += Q[q_elem_idx] * K[k_elem_idx];
        }
        
        score *= scale;
        
        // Apply mask
        if (mask != nullptr) {
            size_t mask_idx = batch_idx * seq_len_q * seq_len_k + q_idx * seq_len_k + k_pos;
            if (mask[mask_idx] == 0) {
                score = -1e9f;
            }
        }
      
        T weight = expf(score - max_score);
        sum_exp += weight;

        size_t v_elem_idx = v_base_offset + k_pos * head_dim + d_idx;
        output_val += weight * V[v_elem_idx];
    }

    output_val /= sum_exp;
    
    size_t out_idx = base_offset + q_idx * head_dim + d_idx;
    Out[out_idx] = output_val;
}

template <typename T>
void launch_flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* Out,
    size_t batch_size, size_t num_heads,
    size_t seq_len_q, size_t seq_len_k, size_t head_dim,
    float scale, const T* mask,
    cudaStream_t stream
) {
    dim3 block(16, 16);  // 16x16 = 256 threads per block
    dim3 grid(
        (seq_len_q + block.x - 1) / block.x,
        (head_dim + block.y - 1) / block.y,
        batch_size * num_heads
    );
    
    flash_attention_kernel<T><<<grid, block, 0, stream>>>(
        Q, K, V, Out,
        batch_size, num_heads,
        seq_len_q, seq_len_k, head_dim,
        scale, mask
    );
    
    CHECK_LAST_CUDA_ERROR();
}

// Explicit template instantiation
template void launch_flash_attention_kernel<float>(
    const float* Q, const float* K, const float* V, float* Out,
    size_t batch_size, size_t num_heads,
    size_t seq_len_q, size_t seq_len_k, size_t head_dim,
    float scale, const float* mask,
    cudaStream_t stream
);
