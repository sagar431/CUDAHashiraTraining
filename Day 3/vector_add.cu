#include <stdio.h>
#include <cuda_runtime.h>

// CPU version
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU version
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20; // 1 million elements
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c_cpu = (float *)malloc(size);
    float *h_c_gpu = (float *)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // CPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("CPU Time: %.2f ms\n", cpu_time);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // GPU timing
    cudaEventRecord(start);
    vectorAddGPU<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %.2f ms\n", gpu_time);

    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < n; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            break;
        }
    }
    printf("Verification passed!\n");

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}