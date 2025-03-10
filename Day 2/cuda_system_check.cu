#include <cuda_runtime.h>
#include <stdio.h>

// Kernel to test basic computation
__global__ void testKernel(float *d_out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    d_out[tid] = float(tid) * 2.0f;
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    printf("\n=== CUDA System Check ===\n\n");

    // 1. Check CUDA devices
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Getting device count");
    printf("Found %d CUDA device(s)\n\n", deviceCount);

    // 2. Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i), "Getting device properties");
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Thread Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    // 3. Test memory allocation and kernel execution
    printf("Testing CUDA operations:\n");
    
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_out = (float*)malloc(size);
    
    // Allocate device memory
    float *d_out;
    printf("- Testing device memory allocation... ");
    checkCudaError(cudaMalloc(&d_out, size), "Device memory allocation");
    printf("SUCCESS\n");

    // Launch kernel
    printf("- Testing kernel launch... ");
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    testKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    printf("SUCCESS\n");

    // Copy results back
    printf("- Testing device to host memory copy... ");
    checkCudaError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "Device to host memory copy");
    printf("SUCCESS\n");

    // Verify results
    printf("- Testing computation results... ");
    bool testPassed = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != float(i) * 2.0f) {
            testPassed = false;
            break;
        }
    }
    printf("%s\n", testPassed ? "SUCCESS" : "FAILED");

    // Cleanup
    cudaFree(d_out);
    free(h_out);

    printf("\nCUDA System Check Complete!\n");
    return 0;
}