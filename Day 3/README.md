# Day 3: GPU vs. CPU Architecture Foundations 🧠

## Overview
On Day 3 of my CUDA learning journey, I explored the fundamental architectural differences between GPUs and CPUs, focusing on how these differences impact parallel programming. I implemented two key examples:

1. **GPU vs. CPU Architecture Comparison** (`gpu_vs_cpu.cu`)
2. **Vector Addition Performance Analysis** (`vector_add.cu`)

## Key Concepts Learned

### Core Architectural Differences

| Feature | CPU | GPU |
|---------|-----|-----|
| **Core Design** | Few complex cores (4-64 typically) | Many simple cores (thousands) |
| **Optimization** | Low-latency serial processing | High-throughput parallel processing |
| **Cache** | Large caches (MB) | Smaller caches (KB) |
| **Control Logic** | Complex branch prediction & out-of-order execution | Simpler in-order execution |
| **Clock Speed** | Higher (3-5 GHz) | Lower (1-2 GHz) |

### Memory Hierarchy Comparison

#### CPU Memory Hierarchy
- **L1 Cache**: 32-64 KB per core, <1ns latency
- **L2 Cache**: 256 KB-1 MB per core, ~3-10ns latency
- **L3 Cache**: 8-64 MB shared, ~10-20ns latency
- **Main Memory**: GB-TB capacity, ~100ns latency

#### GPU Memory Hierarchy
- **Registers**: Per-thread, fastest access
- **Shared Memory**: Per-block, ~5-10ns latency
- **L1/L2 Cache**: Smaller than CPU equivalents
- **Global Memory**: High bandwidth (hundreds of GB/s) but high latency (~200-300ns)

### Execution Models

#### CPU: Complex Serial Processing
- **Superscalar Architecture**: Multiple instructions per cycle
- **Out-of-Order Execution**: Optimizes instruction flow
- **Branch Prediction**: Speculative execution
- **Hyperthreading**: 2-4 threads per core

#### GPU: SIMT (Single Instruction, Multiple Thread)
- **Warps/Wavefronts**: Groups of 32/64 threads executing in lockstep
- **Thread Blocks/Work Groups**: Organized into grids for execution
- **Streaming Multiprocessors (SMs)**: Process multiple warps concurrently
- **Warp Scheduling**: Zero-overhead context switching between warps

## Implementation Examples

### 1. GPU vs. CPU Architecture (`gpu_vs_cpu.cu`)

This program demonstrates:
- Querying and displaying detailed GPU device properties
- Comparing architectural differences between CPU and GPU
- Implementing vector addition on both CPU and GPU
- Measuring and analyzing performance differences
- Explaining how architectural differences impact performance

Key findings:
- GPUs excel at data-parallel tasks with high arithmetic intensity
- CPUs perform better on control-flow heavy tasks with unpredictable branching
- Memory access patterns significantly impact GPU performance

### 2. Vector Addition Performance Analysis (`vector_add.cu`)

This simple example:
- Implements vector addition on both CPU and GPU
- Times the performance of each implementation
- Verifies correctness of results

Key implementation details:
- Uses CUDA events for accurate timing
- Demonstrates proper CUDA memory management (allocation, transfer, freeing)
- Shows basic CUDA kernel structure with thread indexing
- Illustrates grid/block configuration for optimal performance

## Performance Results

For vector addition with 1 million elements:
- **CPU Time**: Varies based on CPU model (typically 10-50ms)
- **GPU Time**: Significantly faster (typically 0.1-1ms)
- **Speedup**: Often 10-100x depending on hardware

## Lessons Learned

1. **Throughput vs. Latency**:
   - GPUs optimize for throughput at the expense of latency
   - CPUs optimize for latency at the expense of throughput

2. **Memory Access Patterns**:
   - Coalesced memory access is critical for GPU performance
   - CPU cache hierarchy benefits different access patterns

3. **Execution Model Impact**:
   - GPU's SIMT model requires careful algorithm design
   - Warp divergence can severely degrade performance

4. **Data Transfer Overhead**:
   - CPU-GPU memory transfers can be a significant bottleneck
   - Must consider data locality and transfer costs in algorithm design

## References

- NVIDIA CUDA C Programming Guide, "Hardware Model Overview"
- NVIDIA GPU Architecture Documentation
- "Professional CUDA C Programming" by John Cheng, Max Grossman, and Ty McKercher

## Next Steps

- Explore shared memory usage for improved performance
- Implement more complex algorithms that leverage GPU architecture
- Investigate warp divergence and its performance impact
- Study memory coalescing techniques for optimal memory access patterns
