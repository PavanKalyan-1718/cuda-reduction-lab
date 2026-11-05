#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (0)

__global__ void reduce_blocks(const float* input, float* partials, std::size_t count) {
  extern __shared__ float shared[];
  const unsigned int thread = threadIdx.x;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + thread;
  float value = 0.0f;
  if (index < count) value += input[index];
  if (index + blockDim.x < count) value += input[index + blockDim.x];
  shared[thread] = value;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) shared[thread] += shared[thread + stride];
    __syncthreads();
  }
  if (thread == 0) partials[blockIdx.x] = shared[0];
}

int main(int argc, char** argv) {
  const std::size_t count = argc > 1 ? std::stoull(argv[1]) : 16'777'216;
  const unsigned int threads = argc > 2 ? std::stoul(argv[2]) : 256;
  if (count == 0 || threads == 0 || (threads & (threads - 1)) != 0) throw std::invalid_argument("count must be positive and threads must be a power of two");

  std::vector<float> host(count);
  for (std::size_t index = 0; index < count; ++index) host[index] = static_cast<float>(index % 1000) / 1000.0f;
  const double expected = [&] { double sum = 0; for (float value : host) sum += value; return sum; }();
  const std::size_t blocks = (count + threads * 2 - 1) / (threads * 2);
  float *device_input = nullptr, *device_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  reduce_blocks<<<blocks, threads, threads * sizeof(float)>>>(device_input, device_partials, count);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::vector<float> partials(blocks); CUDA_CHECK(cudaMemcpy(partials.data(), device_partials, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  double observed = 0; for (float partial : partials) observed += partial;
  const double gib_per_second = (count * sizeof(float)) / (milliseconds / 1000.0) / (1024.0 * 1024.0 * 1024.0);
  std::cout << std::fixed << std::setprecision(3) << "elements=" << count << " threads=" << threads << " blocks=" << blocks << " sum=" << observed << " expected=" << expected << " kernel_ms=" << milliseconds << " effective_GiB_s=" << gib_per_second << "\n";
  CUDA_CHECK(cudaFree(device_input)); CUDA_CHECK(cudaFree(device_partials)); CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return std::abs(observed - expected) < 1e-2 * std::max(1.0, expected) ? 0 : 1;
}

