CXX ?= g++
NVCC ?= nvcc
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: cpu cuda clean

cpu:
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/reduction_cpu.cpp -o build/reduction_cpu

cuda:
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -lineinfo src/reduction_cuda.cu -o build/reduction_cuda

clean:
	rm -rf build

