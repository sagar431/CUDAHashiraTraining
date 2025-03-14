#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define warpSize explicitly since it's causing linker errors
#ifndef warpSize
#define warpSize 32
#endif

// Matrix dimensions
#define N 1024
#define BLOCK_SIZE warpSize  // This is intentionally set to warp size

// Kernel with warp-aware implementation
__global__ void matrixMulOptimized(float *A, float *B, float *C) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread computes one element of C
    float sum = 0.0f;
    
    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Loop over all tiles
    for (int t = 0; t < (N / BLOCK_SIZE); ++t) {
        // Load tiles into shared memory
        As[row][col] = A[(blockRow * BLOCK_SIZE + row) * N + (t * BLOCK_SIZE + col)];
        Bs[row][col] = B[(t * BLOCK_SIZE + row) * N + (blockCol * BLOCK_SIZE + col)];
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Compute partial sum (no divergence in this loop - good for warps)
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[row][k] * Bs[k][col];
        }
        
        // Synchronize to avoid premature loading of next tiles
        __syncthreads();
    }
    
    // Write result
    C[(blockRow * BLOCK_SIZE + row) * N + (blockCol * BLOCK_SIZE + col)] = sum;
}

// Kernel with intentional warp divergence (for comparison)
__global__ void matrixMulDivergent(float *A, float *B, float *C) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Thread ID within block
    int threadId = row * BLOCK_SIZE + col;
    
    // Each thread computes one element of C
    float sum = 0.0f;
    
    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Loop over all tiles
    for (int t = 0; t < (N / BLOCK_SIZE); ++t) {
        // Load tiles into shared memory
        As[row][col] = A[(blockRow * BLOCK_SIZE + row) * N + (t * BLOCK_SIZE + col)];
        Bs[row][col] = B[(t * BLOCK_SIZE + row) * N + (blockCol * BLOCK_SIZE + col)];
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Intentional warp divergence based on thread ID
        // Half the threads in each warp will take one path, half will take another
        if (threadId % 2 == 0) {
            // Even threads
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += As[row][k] * Bs[k][col];
            }
        } else {
            // Odd threads
            for (int k = BLOCK_SIZE - 1; k >= 0; --k) {
                sum += As[row][k] * Bs[k][col];
            }
        }
        
        // Synchronize to avoid premature loading of next tiles
        __syncthreads();
    }
    
    // Write result
    C[(blockRow * BLOCK_SIZE + row) * N + (blockCol * BLOCK_SIZE + col)] = sum;
}

// Initialize matrices with random values
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Verify results against CPU implementation
bool verifyResults(float *A, float *B, float *C, int size) {
    float *verifyC = (float*)malloc(size * size * sizeof(float));
    
    // CPU matrix multiplication
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            verifyC[i * size + j] = sum;
        }
    }
    
    // Compare results
    bool correct = true;
    for (int i = 0; i < size * size; i++) {
        // Allow for small floating point differences
        if (fabs(C[i] - verifyC[i]) > 1e-5) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, C[i], verifyC[i]);
            correct = false;
            break;
        }
    }
    
    free(verifyC);
    return correct;
}

int main() {
    // Set random seed
    srand(time(NULL));
    
    // Allocate host memory
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C_optimized = (float*)malloc(N * N * sizeof(float));
    float *h_C_divergent = (float*)malloc(N * N * sizeof(float));
    
    // Initialize matrices
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    printf("Matrix size: %d x %d\n", N, N);
    printf("Block size: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Grid size: %d x %d\n\n", N / BLOCK_SIZE, N / BLOCK_SIZE);
    
    // Run optimized kernel
    printf("Running warp-optimized kernel...\n");
    cudaEventRecord(start);
    matrixMulOptimized<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C_optimized, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Optimized kernel execution time: %.3f ms\n", milliseconds);
    printf("Verification: %s\n\n", verifyResults(h_A, h_B, h_C_optimized, N) ? "PASSED" : "FAILED");
    
    // Run divergent kernel
    printf("Running warp-divergent kernel...\n");
    cudaEventRecord(start);
    matrixMulDivergent<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C_divergent, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Divergent kernel execution time: %.3f ms\n", milliseconds);
    printf("Verification: %s\n\n", verifyResults(h_A, h_B, h_C_divergent, N) ? "PASSED" : "FAILED");
    
    // Calculate performance difference
    float speedup = milliseconds / milliseconds;
    printf("Performance impact of warp divergence: %.2fx slower\n", speedup);
    
    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_optimized);
    free(h_C_divergent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
