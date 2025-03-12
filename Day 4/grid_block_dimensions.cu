#include <stdio.h>

/*
 * This kernel demonstrates thread indexing across different grid and block dimensions
 * Each thread will print its unique coordinates and global index
 */
__global__ void threadIndexing() {
    // Calculate the global thread ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For 2D grids, calculate row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For 3D calculations
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Print thread information
    printf("Thread: [%d, %d, %d] in Block: [%d, %d, %d] - Global ID: %d - 2D Position: (%d, %d) - 3D Position: (%d, %d, %d)\n", 
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadId,
           col, row,
           col, row, depth);
}

int main() {
    // Configuration 1: 1D Grid of 1D Blocks
    printf("\n=== Configuration 1: 1D Grid of 1D Blocks ===\n");
    dim3 blocks1D(2);         // 2 blocks in x dimension
    dim3 threads1D(4);        // 4 threads per block in x dimension
    threadIndexing<<<blocks1D, threads1D>>>();
    cudaDeviceSynchronize();
    
    // Configuration 2: 1D Grid of 2D Blocks
    printf("\n=== Configuration 2: 1D Grid of 2D Blocks ===\n");
    dim3 blocks1D_2(2);       // 2 blocks in x dimension
    dim3 threads2D(2, 2);     // 2x2 threads per block
    threadIndexing<<<blocks1D_2, threads2D>>>();
    cudaDeviceSynchronize();
    
    // Configuration 3: 2D Grid of 1D Blocks
    printf("\n=== Configuration 3: 2D Grid of 1D Blocks ===\n");
    dim3 blocks2D(2, 2);      // 2x2 grid of blocks
    dim3 threads1D_2(3);      // 3 threads per block in x dimension
    threadIndexing<<<blocks2D, threads1D_2>>>();
    cudaDeviceSynchronize();
    
    // Configuration 4: 2D Grid of 2D Blocks
    printf("\n=== Configuration 4: 2D Grid of 2D Blocks ===\n");
    dim3 blocks2D_2(2, 2);    // 2x2 grid of blocks
    dim3 threads2D_2(2, 2);   // 2x2 threads per block
    threadIndexing<<<blocks2D_2, threads2D_2>>>();
    cudaDeviceSynchronize();
    
    // Configuration 5: 3D Grid of 3D Blocks
    printf("\n=== Configuration 5: 3D Grid of 3D Blocks ===\n");
    dim3 blocks3D(2, 2, 2);   // 2x2x2 grid of blocks
    dim3 threads3D(2, 2, 2);  // 2x2x2 threads per block
    threadIndexing<<<blocks3D, threads3D>>>();
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
    
    return 0;
}
