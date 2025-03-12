#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

// Array size
#define N 10000000

// Kernel with one element per thread
__global__ void vectorAdd_onePerThread(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Kernel with two elements per thread (thread coarsening)
__global__ void vectorAdd_twoPerThread(const float *A, const float *B, float *C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (i < n) {
        C[i] = A[i] + B[i];
    }
    
    if (i + 1 < n) {
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

// Kernel with four elements per thread (more thread coarsening)
__global__ void vectorAdd_fourPerThread(const float *A, const float *B, float *C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    for (int j = 0; j < 4 && i + j < n; j++) {
        C[i + j] = A[i + j] + B[i + j];
    }
}

// Function to measure kernel execution time
float measureKernelTime(void (*launchKernel)(const float*, const float*, float*, int, int), 
                       const float *d_A, const float *d_B, float *d_C, int n, int blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    launchKernel(d_A, d_B, d_C, n, blockSize);
    
    // Measure execution time
    cudaEventRecord(start);
    launchKernel(d_A, d_B, d_C, n, blockSize);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// Wrapper functions for kernel launches
void launch_onePerThread(const float *d_A, const float *d_B, float *d_C, int n, int blockSize) {
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd_onePerThread<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
}

void launch_twoPerThread(const float *d_A, const float *d_B, float *d_C, int n, int blockSize) {
    int numBlocks = (n / 2 + blockSize - 1) / blockSize;
    vectorAdd_twoPerThread<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
}

void launch_fourPerThread(const float *d_A, const float *d_B, float *d_C, int n, int blockSize) {
    int numBlocks = (n / 4 + blockSize - 1) / blockSize;
    vectorAdd_fourPerThread<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Vector size: %d elements\n\n", N);
    
    // Test different block sizes
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    printf("%-15s %-15s %-15s %-15s\n", "Block Size", "1 Element/Thread", "2 Elements/Thread", "4 Elements/Thread");
    printf("----------------------------------------------------------------\n");
    
    for (int i = 0; i < numBlockSizes; i++) {
        int blockSize = blockSizes[i];
        
        float time1 = measureKernelTime(launch_onePerThread, d_A, d_B, d_C, N, blockSize);
        float time2 = measureKernelTime(launch_twoPerThread, d_A, d_B, d_C, N, blockSize);
        float time4 = measureKernelTime(launch_fourPerThread, d_A, d_B, d_C, N, blockSize);
        
        printf("%-15d %-15.3f %-15.3f %-15.3f\n", blockSize, time1, time2, time4);
    }
    
    // Verify results
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
