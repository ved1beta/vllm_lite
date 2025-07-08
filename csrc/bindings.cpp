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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_launcher", &gemm_launcher, "GEMM kernel launcher (CUDA)");
}