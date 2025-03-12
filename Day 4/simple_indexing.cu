#include <stdio.h>

// Simple kernel to demonstrate thread indexing
__global__ void printThreadIndex() {
    // Calculate thread indices
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print thread information
    printf("Thread [%d, %d] in Block [%d, %d] - Global Index: %d\n",
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, globalIdx);
}

int main() {
    // Define grid and block dimensions
    dim3 blockDim(4, 1);  // 4 threads per block in x dimension
    dim3 gridDim(2, 1);   // 2 blocks in x dimension
    
    printf("=== 1D Grid of 1D Blocks ===\n");
    printf("Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("Total threads: %d\n\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    // Launch kernel
    printThreadIndex<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    // Try a 2D grid
    dim3 blockDim2(2, 2);  // 2x2 threads per block
    dim3 gridDim2(2, 2);   // 2x2 grid of blocks
    
    printf("\n=== 2D Grid of 2D Blocks ===\n");
    printf("Grid dimensions: (%d, %d)\n", gridDim2.x, gridDim2.y);
    printf("Block dimensions: (%d, %d)\n", blockDim2.x, blockDim2.y);
    printf("Total threads: %d\n\n", gridDim2.x * gridDim2.y * blockDim2.x * blockDim2.y);
    
    // Launch kernel with 2D configuration
    printThreadIndex<<<gridDim2, blockDim2>>>();
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
    
    return 0;
}
