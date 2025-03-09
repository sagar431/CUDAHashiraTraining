#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 1000000;  // Vector size
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a failed");
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b failed");
    checkCudaError(cudaMalloc(&d_c, size), "cudaMalloc d_c failed");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), 
                  "cudaMemcpy H2D d_a failed");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), 
                  "cudaMemcpy H2D d_b failed");

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), 
                  "cudaMemcpy D2H failed");

    // Verify results
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Vector addition completed successfully!\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
} 