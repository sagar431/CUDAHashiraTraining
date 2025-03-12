#include <stdio.h>

// Matrix dimensions
#define N 8

// Kernel for matrix addition using 1D grid and 1D blocks
__global__ void matrixAdd1D(int *a, int *b, int *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel for matrix addition using 2D grid and 2D blocks
__global__ void matrixAdd2D(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int idx = row * N + col;
        c[idx] = a[idx] + b[idx];
    }
}

// Utility function to initialize a matrix
void initMatrix(int *mat) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = rand() % 10;
    }
}

// Utility function to print a matrix
void printMatrix(int *mat, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%2d ", mat[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Allocate host memory
    int *h_a, *h_b, *h_c1D, *h_c2D;
    h_a = (int*)malloc(N * N * sizeof(int));
    h_b = (int*)malloc(N * N * sizeof(int));
    h_c1D = (int*)malloc(N * N * sizeof(int));
    h_c2D = (int*)malloc(N * N * sizeof(int));
    
    // Initialize matrices
    srand(42); // For reproducible results
    initMatrix(h_a);
    initMatrix(h_b);
    
    // Allocate device memory
    int *d_a, *d_b, *d_c1D, *d_c2D;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c1D, N * N * sizeof(int));
    cudaMalloc(&d_c2D, N * N * sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch 1D kernel
    printf("=== Matrix Addition using 1D Grid and 1D Blocks ===\n");
    int blockSize1D = 16;
    int numBlocks1D = (N * N + blockSize1D - 1) / blockSize1D;
    
    printf("Grid configuration: %d blocks, %d threads per block\n", 
           numBlocks1D, blockSize1D);
    
    matrixAdd1D<<<numBlocks1D, blockSize1D>>>(d_a, d_b, d_c1D);
    cudaDeviceSynchronize();
    
    // Launch 2D kernel
    printf("\n=== Matrix Addition using 2D Grid and 2D Blocks ===\n");
    dim3 blockSize2D(4, 4);  // 4x4 threads per block
    dim3 numBlocks2D((N + blockSize2D.x - 1) / blockSize2D.x, 
                     (N + blockSize2D.y - 1) / blockSize2D.y);
    
    printf("Grid configuration: (%d, %d) blocks, (%d, %d) threads per block\n", 
           numBlocks2D.x, numBlocks2D.y, blockSize2D.x, blockSize2D.y);
    
    matrixAdd2D<<<numBlocks2D, blockSize2D>>>(d_a, d_b, d_c2D);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_c1D, d_c1D, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c2D, d_c2D, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print matrices
    printMatrix(h_a, "Matrix A");
    printMatrix(h_b, "Matrix B");
    printMatrix(h_c1D, "Result using 1D approach");
    printMatrix(h_c2D, "Result using 2D approach");
    
    // Verify results match
    bool match = true;
    for (int i = 0; i < N * N; i++) {
        if (h_c1D[i] != h_c2D[i]) {
            match = false;
            printf("Mismatch at index %d: 1D result = %d, 2D result = %d\n", 
                   i, h_c1D[i], h_c2D[i]);
            break;
        }
    }
    
    if (match) {
        printf("Results from 1D and 2D approaches match!\n");
    }
    
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c1D);
    free(h_c2D);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c1D);
    cudaFree(d_c2D);
    
    return 0;
}
