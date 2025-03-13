# Simple Vector Addition with Triton on GPU

## Overview
This project demonstrates a simple vector addition using Triton, a tool for writing GPU code in Python. It adds two lists of numbers (vectors) using the GPU, which is faster than a CPU for large lists. This is my first attempt at learning CUDA and Triton, and I'm sharing it to document my progress.

## What This Code Does
- Adds two lists of numbers (`x` and `y`) using the GPU.
- Uses Triton to write GPU instructions (called a "kernel").
- Tests the results to make sure they're correct.
- Runs on Google Colab with a GPU.

## Why I Made This
- I'm new to CUDA and Triton, and I wanted to understand how GPUs work.
- This is a simplified version of vector addition to help me learn the basics.
- I used small lists (size=10) to make it easier to understand.

## How to Run
1. Open Google Colab (https://colab.research.google.com/).
2. Select a GPU runtime:
   - Click "Runtime" > "Change runtime type" > Select "GPU" > Save.
3. Copy the code from `vector_add.py` into a Colab cell.
4. Run the cell. It will:
   - Install Triton.
   - Add two random lists of 10 numbers.
   - Print the inputs and results.
   - Check if the results are correct.

## Code Explanation
### `vector_add.py`
- **Install Triton**: Installs the Triton library (only needed once in Colab).
- **Imports**:
  - `torch`: For creating lists (tensors) and checking results.
  - `triton`: For writing GPU code.
  - `triton.language as tl`: For special Triton operations.
- **Set Device**: Ensures everything runs on the GPU.
- **Kernel (`add_kernel`)**:
  - Instructions for GPU workers to add numbers.
  - Each group of workers (block) handles 4 numbers (`BLOCK_SIZE=4`).
  - Splits the work into groups based on list size.
- **Wrapper (`add`)**:
  - Sets up the lists and launches the kernel.
  - Ensures all lists are on the GPU.
- **Test (`test_add`)**:
  - Creates two random lists of 10 numbers.
  - Runs the GPU addition and PyTorch's addition (for comparison).
  - Prints the inputs and results.
  - Checks if the results match.

### Example Outpu