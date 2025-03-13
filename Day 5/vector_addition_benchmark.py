import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import triton
import triton.language as tl

# Import the Triton vector addition from our previous file
from vector_addition import vector_add as triton_vector_add

# CUDA implementation for comparison
cuda_code = """
extern "C" __global__ void vector_add_cuda(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Load CUDA kernel
cuda_kernel = None

def load_cuda_kernel():
    global cuda_kernel
    if cuda_kernel is None:
        cuda_kernel = torch.utils.cpp_extension.load_inline(
            name="vector_add_cuda",
            cpp_sources="",
            cuda_sources=cuda_code,
            functions=["vector_add_cuda"],
            with_cuda=True,
        )
    return cuda_kernel

def cuda_vector_add(a, b):
    """
    Compute c = a + b using CUDA
    """
    # Check input dimensions
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    # Output tensor
    c = torch.empty_like(a)
    
    # Get number of elements
    n_elements = a.numel()
    
    # Define block size
    block_size = 1024
    grid_size = (n_elements + block_size - 1) // block_size
    
    # Load CUDA kernel
    kernel = load_cuda_kernel()
    
    # Launch kernel
    kernel.vector_add_cuda(
        grid=grid_size,
        block=block_size,
        args=[a, b, c, n_elements],
    )
    
    return c

def benchmark(sizes):
    """
    Benchmark Triton vs CUDA vs PyTorch for vector addition
    """
    triton_times = []
    cuda_times = []
    pytorch_times = []
    
    for size in sizes:
        # Create input vectors
        a_cpu = torch.rand(size, dtype=torch.float32)
        b_cpu = torch.rand(size, dtype=torch.float32)
        
        # Move to GPU
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warm up
        _ = triton_vector_add(a_gpu, b_gpu)
        try:
            _ = cuda_vector_add(a_gpu, b_gpu)
        except:
            print("CUDA kernel compilation failed. Skipping CUDA benchmark.")
            cuda_times = None
        _ = a_gpu + b_gpu
        
        # Measure Triton performance
        torch.cuda.synchronize()
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            _ = triton_vector_add(a_gpu, b_gpu)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / iterations
        triton_times.append(triton_time * 1000)  # Convert to ms
        
        # Measure CUDA performance
        if cuda_times is not None:
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = cuda_vector_add(a_gpu, b_gpu)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start) / iterations
            cuda_times.append(cuda_time * 1000)  # Convert to ms
        
        # Measure PyTorch performance
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = a_gpu + b_gpu
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / iterations
        pytorch_times.append(pytorch_time * 1000)  # Convert to ms
        
        print(f"Size: {size}")
        print(f"  Triton: {triton_time*1000:.3f} ms")
        if cuda_times is not None:
            print(f"  CUDA: {cuda_time*1000:.3f} ms")
        print(f"  PyTorch: {pytorch_time*1000:.3f} ms")
    
    return triton_times, cuda_times, pytorch_times

def plot_results(sizes, triton_times, cuda_times, pytorch_times):
    """
    Plot benchmark results
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, triton_times, 'o-', label='Triton')
    if cuda_times is not None:
        plt.plot(sizes, cuda_times, 's-', label='CUDA')
    plt.plot(sizes, pytorch_times, '^-', label='PyTorch')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Vector Size')
    plt.ylabel('Time (ms)')
    plt.title('Vector Addition Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('vector_addition_benchmark.png')
    plt.close()

def main():
    # Vector sizes to benchmark
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    
    # Run benchmark
    triton_times, cuda_times, pytorch_times = benchmark(sizes)
    
    # Plot results
    try:
        plot_results(sizes, triton_times, cuda_times, pytorch_times)
        print("Benchmark plot saved as 'vector_addition_benchmark.png'")
    except Exception as e:
        print(f"Failed to create plot: {e}")

if __name__ == "__main__":
    main()
