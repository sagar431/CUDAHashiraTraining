# Day 4: CUDA Grid and Block Dimensions

## Overview

In Day 4, we explore how to launch CUDA kernels with different grid and block dimensions and understand thread indexing. This is a critical concept in CUDA programming as it determines how your computation is mapped to the GPU's execution model.

## Key Concepts

### Grid and Block Dimensions

In CUDA, work is organized in a hierarchical structure:
- **Thread**: The basic unit of execution
- **Block**: A group of threads that can cooperate via shared memory and synchronization
- **Grid**: A collection of blocks that execute the same kernel

Dimensions can be specified in 1D, 2D, or 3D:
```cpp
// 1D configuration
dim3 blocks1D(N);         // N blocks in x dimension
dim3 threads1D(M);        // M threads per block in x dimension

// 2D configuration
dim3 blocks2D(N, M);      // NxM grid of blocks
dim3 threads2D(P, Q);     // PxQ threads per block

// 3D configuration
dim3 blocks3D(N, M, K);   // NxMxK grid of blocks
dim3 threads3D(P, Q, R);  // PxQxR threads per block
```

### Thread Indexing

Each thread has a unique index that can be calculated from:
- `threadIdx`: Position within a block (x, y, z)
- `blockIdx`: Position of the block within the grid (x, y, z)
- `blockDim`: Dimensions of a block (x, y, z)
- `gridDim`: Dimensions of the grid (x, y, z)

For 1D indexing:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D indexing:
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;  // For a row-major layout
```

For 3D indexing:
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;  // For a row-major layout
```

## Code Examples

### 1. Grid and Block Dimensions Demonstration (`grid_block_dimensions.cu`)

This program demonstrates various grid and block configurations (1D, 2D, and 3D) and shows how thread indexing works in each case.

To compile and run:
```bash
nvcc -o grid_block_dimensions grid_block_dimensions.cu
./grid_block_dimensions
```

### 2. Matrix Operations (`matrix_operations.cu`)

This example shows how to implement matrix addition using both 1D and 2D thread organizations. It demonstrates:
- How to map a 2D problem (matrix) to both 1D and 2D thread organizations
- How to calculate the correct indices for each approach
- How to choose appropriate block sizes

To compile and run:
```bash
nvcc -o matrix_operations matrix_operations.cu
./matrix_operations
```

### 3. Thread Coarsening (`thread_coarsening.cu`)

This advanced example demonstrates thread coarsening - a technique where each thread processes multiple elements instead of just one. It shows:
- Performance implications of different thread workloads
- How block size affects performance
- How to measure kernel execution time

To compile and run:
```bash
nvcc -o thread_coarsening thread_coarsening.cu
./thread_coarsening
```

## Best Practices

1. **Choose appropriate dimensions**: Match your problem structure (1D for vectors, 2D for matrices, etc.)
2. **Block size considerations**: 
   - Typically a multiple of 32 (warp size)
   - Common values: 128, 256, 512 threads per block
   - Too small: underutilizes GPU
   - Too large: limits occupancy
3. **Thread coarsening**: Sometimes having each thread do more work is more efficient
4. **Memory access patterns**: Align thread indices with memory access patterns for coalescing

## Exercises

1. Modify the `grid_block_dimensions.cu` program to use different grid and block dimensions and observe how the thread indices change.
2. Implement matrix multiplication using both 1D and 2D thread organizations.
3. Experiment with the `thread_coarsening.cu` program using different vector sizes and block dimensions.
4. Create a program that processes a 2D image using different block sizes and measure the performance differences.
