#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Size of the arrays to process
#define N 10000000

// CUDA kernel for vector addition
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// CPU function for vector addition
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Function to print device properties
void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("\n=== GPU Architecture Information ===\n");
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    }
    
    printf("\n=== CPU vs. GPU Architectural Comparison ===\n");
    printf("CPU Cores:\n");
    printf("  - Designed for low-latency serial processing\n");
    printf("  - Complex control logic and large caches\n");
    printf("  - Optimized for single-thread performance\n");
    printf("  - Typically few cores (4-64 on consumer hardware)\n\n");
    
    printf("GPU Streaming Multiprocessors (SMs):\n");
    printf("  - Designed for high-throughput parallel processing\n");
    printf("  - Simple control logic with smaller caches\n");
    printf("  - Optimized for data-parallel workloads\n");
    printf("  - Many cores (thousands) organized in SMs\n");
    printf("  - Uses SIMT (Single Instruction, Multiple Thread) execution model\n");
}

int main() {
    // Print device properties and architectural comparison
    printDeviceProperties();
    
    // Allocate host memory
    size_t size = N * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_gpu = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size), "Allocating device memory for d_a");
    checkCudaError(cudaMalloc(&d_b, size), "Allocating device memory for d_b");
    checkCudaError(cudaMalloc(&d_c, size), "Allocating device memory for d_c");
    
    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Copying h_a to d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Copying h_b to d_b");
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("\n=== Performance Comparison ===\n");
    printf("Array size: %d elements\n", N);
    printf("Grid dimensions: %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // GPU Computation with timing
    printf("\nRunning on GPU...\n");
    cudaEventRecord(start);
    
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    checkCudaError(cudaGetLastError(), "Launching kernel");
    
    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "Waiting for kernel to finish");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuMilliseconds = 0;
    cudaEventElapsedTime(&gpuMilliseconds, start, stop);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost), "Copying d_c to h_c_gpu");
    
    // CPU Computation with timing
    printf("Running on CPU...\n");
    clock_t cpu_start = clock();
    
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    
    clock_t cpu_end = clock();
    float cpuMilliseconds = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-5) {
            correct = false;
            printf("Results don't match at index %d! GPU: %f, CPU: %f\n", 
                   i, h_c_gpu[i], h_c_cpu[i]);
            break;
        }
    }
    
    if (correct) {
        printf("Results match! Both implementations are correct.\n\n");
        
        // Print performance comparison
        printf("Performance Results:\n");
        printf("  GPU Time: %.3f ms\n", gpuMilliseconds);
        printf("  CPU Time: %.3f ms\n", cpuMilliseconds);
        printf("  Speedup: %.2fx\n\n", cpuMilliseconds / gpuMilliseconds);
        
        printf("=== Architectural Impact Analysis ===\n");
        printf("1. Throughput vs. Latency:\n");
        printf("   - GPU: High throughput design with many simple cores (%.2fx faster)\n", cpuMilliseconds / gpuMilliseconds);
        printf("   - CPU: Low latency design with few complex cores\n\n");
        
        printf("2. Memory Access Patterns:\n");
        printf("   - GPU: Coalesced memory access is critical for performance\n");
        printf("   - CPU: Cache hierarchy optimized for temporal and spatial locality\n\n");
        
        printf("3. Execution Model:\n");
        printf("   - GPU: SIMT (Single Instruction, Multiple Thread)\n");
        printf("   - CPU: Primarily sequential with some SIMD capabilities\n\n");
        
        printf("4. Ideal Workloads:\n");
        printf("   - GPU: Data-parallel tasks with high arithmetic intensity\n");
        printf("   - CPU: Control-flow heavy tasks with unpredictable branching\n");
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);
    
    return 0;
}