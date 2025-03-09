# Day 1: Introduction to GPU Computing & CUDA 🔥

## **Core Topic** 📚
Introduction to GPU computing and NVIDIA CUDA. Understanding the basics of parallel programming and the role of GPUs in accelerating computations - like mastering the fundamentals of Total Concentration Breathing in Demon Slayer!

## **Practical Exercise / Mini-Project** ⚔️
- **Task**: Write a simple "Hello GPU" kernel that prints a message from one thread
- **Code**: See `./code/hello_gpu.cu` for the implementation
- **Steps**:
  1. Launch a kernel with a single thread (like a lone Demon Slayer on patrol)
  2. Use `printf` to print a message from the GPU
  3. Compile and run using `nvcc`

## **Code Example** 💻
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU() {
    printf("Greetings from Thread %d - Ready to slay some demons!\n", 
           threadIdx.x);
}

int main() {
    // Launch the kernel with 1 thread
    helloGPU<<<1, 1>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}
```

## **Debugging Pitfalls** 🎯
- **Common Issues**:
  - Missing `<cuda_runtime.h>` in the source file
  - Incorrect compiler flags (e.g., forgetting `-arch=sm_XX`)
  - Not checking CUDA error codes
- **Solutions**:
  - Ensure CUDA Toolkit is installed and headers are included
  - Verify NVCC flags in the compilation command
  - Always check return values from CUDA API calls

## **Lessons Learned** 📝
- GPUs are designed for massive parallelism, unlike CPUs
- CUDA kernels are executed by threads organized in grids and blocks
- Think of CUDA threads like Demon Slayer corps members:
  - Each thread has its own task (like individual Demon Slayers)
  - They work together in blocks (like Demon Slayer squads)
  - Multiple blocks form a grid (like the entire Demon Slayer corps)

## **Resource Suggestions** 📖
- CUDA C Programming Guide: Introduction section
- NVIDIA CUDA Samples: "Hello World" example
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)

## **Notes** 📌
- See `./notes/pitfalls.md` for detailed notes on debugging issues
- Next steps: Set up the development environment (Day 2)

## **Today's Achievement** 🏆
Just as Tanjiro took his first steps in Demon Slayer training, we've taken our first steps in CUDA programming. Remember: every master was once a beginner! 