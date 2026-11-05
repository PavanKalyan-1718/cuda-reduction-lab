# CUDA Reduction Lab

An original C++/CUDA benchmark for studying a foundational parallel primitive: reducing a large vector to a scalar.

It includes a portable CPU reference executable and a CUDA implementation with a shared-memory block reduction. The goal is to make correctness, bandwidth, and launch configuration explicit before comparing kernels.

## Run the CPU reference

```bash
make cpu
./build/reduction_cpu 1000000
```

The program checks the computed reduction against the closed-form sum for the generated input and reports elapsed time and effective memory bandwidth.

## Run the CUDA kernel

Requires the CUDA toolkit and a compatible GPU.

```bash
make cuda
./build/reduction_cuda 16777216 256
```

The CUDA executable verifies the GPU result against a CPU reference and prints kernel time, bandwidth, and configuration.

## Why this exists

Reduction is a useful performance-engineering exercise because it exposes memory coalescing, occupancy, synchronization, launch configuration, numerical correctness, and the gap between a reference implementation and a parallel implementation.

## Project layout

```text
src/reduction_cpu.cpp   Portable correctness and bandwidth baseline
src/reduction_cuda.cu   Shared-memory CUDA reduction implementation
Makefile                Separate CPU and CUDA build targets
```

