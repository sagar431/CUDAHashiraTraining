# CUDA Thread Hierarchy: Understanding Warps with Matrix Multiplication

## Overview
This module explores CUDA's thread execution model, focusing on warps - the fundamental execution units in NVIDIA GPUs. We'll demonstrate warp behavior through matrix multiplication, a common GPU workload.

## What are Warps?

A warp is a group of 32 threads that execute the same instruction simultaneously on different data (SIMD - Single Instruction, Multiple Data). Key points about warps:

- **Fixed Size**: On all current NVIDIA GPUs, a warp consists of exactly 32 threads
- **Execution Unit**: The warp is the smallest execution unit in CUDA
- **Thread Block Division**: Each thread block is divided into warps
- **Lockstep Execution**: All threads in a warp execute the same instruction at the same time

## Thread Hierarchy Recap

CUDA organizes threads in a hierarchical structure:
1. **Grid**: Collection of thread blocks
2. **Block**: Group of threads that can cooperate via shared memory
3. **Warp**: Group of 32 threads that execute together
4. **Thread**: Individual execution unit

```
Grid
├── Block (0,0)    Block (1,0)    ...
│    ├── Warp 0    ├── Warp 0
│    ├── Warp 1    ├── Warp 1
│    └── ...       └── ...
└── ...
```

## Warp Divergence

One of the most critical concepts to understand about warps is **warp divergence**:

- **Definition**: When threads within the same warp take different execution paths due to conditional statements
- **Performance Impact**: Significant performance penalty as the warp must execute all paths serially
- **Example**: If half the threads in a warp execute the "if" branch and half execute the "else" branch, execution time doubles

```cuda
if (threadIdx.x % 2 == 0) {
    // Path A - executed by even-numbered threads
} else {
    // Path B - executed by odd-numbered threads
}
```

In the example above, the warp will execute both paths serially, effectively halving performance.

## Matrix Multiplication Example

Our matrix multiplication example demonstrates:

1. **Optimized Implementation**: All threads in a warp follow the same execution path
2. **Divergent Implementation**: Threads in the same warp take different paths based on thread ID
3. **Performance Comparison**: Shows the impact of warp divergence on real-world computation

### Key Optimization Techniques

- **Block Size = Warp Size**: Using a block size that's a multiple of the warp size (32)
- **Shared Memory**: Using shared memory to reduce global memory access
- **Coalesced Memory Access**: Ensuring threads in a warp access adjacent memory locations
- **Avoiding Divergence**: Structuring the computation to minimize different execution paths

## Compiling and Running

To compile and run the matrix multiplication example:

```bash
cd /home/sagar/Desktop/CUDAHashiraTraining/Day\ 6/
nvcc -o warp_matrix_multiplication warp_matrix_multiplication.cu -lcudart
./warp_matrix_multiplication
```

The output will show the performance difference between the optimized and divergent implementations.
