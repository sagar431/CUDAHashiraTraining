#include <cuda_runtime.h>
#include <stdio.h>

// cuda kernel to print message from thread 
__global__ void helloGPU(){
    printf("Hello from gpu thread %d! \n", threadIdx.x);
}

int main(){
    // Check if CUDA device is available
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Using GPU: %s\n", prop.name);

    // Try to initialize the primary device
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // launch the kernel with 1 block and 1 thread 
    helloGPU<<<1,1>>>();
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // synchronize and flush output
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Synchronization error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    fflush(stdout);

    printf("Hello from cpu!\n");
    fflush(stdout);

    return 0;
}