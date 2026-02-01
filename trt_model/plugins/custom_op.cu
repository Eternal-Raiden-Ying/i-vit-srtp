#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// CUDA kernel: 实际执行 INT32 -> INT8 的逐元素重标定
__global__ void requant_int32_to_int8_kernel(
    const int32_t* input,
    int8_t* output,
    int64_t n,
    float scale1,
    float scale2
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = static_cast<float>(input[idx]) * scale1 / scale2;
    int32_t q = static_cast<int32_t>(lrintf(x));
    q = q > 127 ? 127 : q;
    q = q < -128 ? -128 : q;
    output[idx] = static_cast<int8_t>(q);
}

// C++ 插件调用的入口（host function），内部 launch kernel
extern "C" void requant_int32_to_int8_cuda(
    const int32_t* input,
    int8_t* output,
    int64_t n,
    float scale1,
    float scale2,
    cudaStream_t stream
) {
    if (n <= 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    requant_int32_to_int8_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n, scale1, scale2
    );
}
