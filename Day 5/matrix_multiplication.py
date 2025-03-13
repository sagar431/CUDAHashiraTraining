import torch
import triton
import triton.language as tl
import time

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing C = A @ B where A, B, and C are matrices.
    A has shape (M, K), B has shape (K, N), and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ID to the block of C it should compute
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the current block
    # A block pointer has shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # B block pointer has shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # ----------------------------------------------------------
    # Offset from the start of the matrix to the start of the block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create a mask to handle the case where the block size doesn't divide the matrix size
    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N
    
    # Pointers to the block of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `acc` variable, and then write back to C
    # -----------------------------------------------------------
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, using masks to handle boundary conditions
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        
        # Load A and B blocks from DRAM
        a = tl.load(a_ptrs, mask=a_mask & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & b_mask, other=0.0)
        
        # We compute a block of the C matrix by taking the dot product
        # of a block of A and a block of B
        acc += tl.dot(a, b)
        
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # -----------------------------------------------------------
    # Write back the result
    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul(a, b):
    """
    Compute C = A @ B using Triton
    """
    # Check input dimensions
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} and {b.shape}"
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    
    # Get dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Set block sizes (these can be tuned)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Get strides for accessing tensors
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)
    
    # Calculate grid dimensions
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c

def main():
    # Matrix dimensions
    M = 1024
    K = 1024
    N = 1024
    
    # Create input matrices on CPU
    a_cpu = torch.rand((M, K), dtype=torch.float32)
    b_cpu = torch.rand((K, N), dtype=torch.float32)
    
    # Move to GPU
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    
    # Compute using Triton
    c_triton = matmul(a_gpu, b_gpu)
    
    # Compute using PyTorch for verification
    c_torch = torch.matmul(a_gpu, b_gpu)
    
    # Verify results
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-2, atol=1e-2)
    print("✓ Triton and PyTorch results match!")
    
    # Performance measurement
    # Warm up
    for _ in range(10):
        _ = matmul(a_gpu, b_gpu)
        _ = torch.matmul(a_gpu, b_gpu)
    
    # Measure Triton performance
    torch.cuda.synchronize()
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        _ = matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    # Measure PyTorch performance
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # Print performance results
    print(f"Matrix dimensions: {M}x{K} @ {K}x{N}")
    print(f"Triton: {triton_time*1000:.3f} ms")
    print(f"PyTorch: {torch_time*1000:.3f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

if __name__ == "__main__":
    main()
