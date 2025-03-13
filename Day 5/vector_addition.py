import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    n_elements,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes c = a + b using Triton
    """
    # Program ID
    pid = tl.program_id(axis=0)
    # Block start
    block_start = pid * BLOCK_SIZE
    # Offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle case where the block size doesn't divide the number of elements
    mask = offsets < n_elements
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # Compute
    c = a + b
    # Store result
    tl.store(c_ptr + offsets, c, mask=mask)

def vector_add(a, b):
    """
    Compute c = a + b using Triton
    """
    # Check input dimensions
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    # Output tensor
    c = torch.empty_like(a)
    
    # Get number of elements
    n_elements = a.numel()
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vector_add_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE,
    )
    
    return c

def main():
    # Vector size
    N = 1_000_000
    
    # Create input vectors on CPU
    a_cpu = torch.rand(N, dtype=torch.float32)
    b_cpu = torch.rand(N, dtype=torch.float32)
    
    # Move to GPU
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    
    # Compute using Triton
    c_triton = vector_add(a_gpu, b_gpu)
    
    # Compute using PyTorch for verification
    c_torch = a_gpu + b_gpu
    
    # Verify results
    torch.testing.assert_close(c_triton, c_torch)
    print("✓ Triton and PyTorch results match!")
    
    # Performance measurement
    import time
    
    # Warm up
    for _ in range(10):
        _ = vector_add(a_gpu, b_gpu)
    
    # Measure Triton performance
    torch.cuda.synchronize()
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = vector_add(a_gpu, b_gpu)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    # Measure PyTorch performance
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = a_gpu + b_gpu
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # Print performance results
    print(f"Vector size: {N}")
    print(f"Triton: {triton_time*1000:.3f} ms")
    print(f"PyTorch: {torch_time*1000:.3f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

if __name__ == "__main__":
    main()
