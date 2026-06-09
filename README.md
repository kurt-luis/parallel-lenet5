# Parallel Implementations of the LeNet-5 Architecture on the CIFAR-10 Dataset

> **CS 171 Mini-Project** | Kurt Luis S. Landoy | Electrical and Electronics Engineering Institute, University of the Philippines Diliman

A from-scratch white-box implementation and hardware analysis of the LeNet-5 forward pass across three distinct computational paradigms: a sequential **CPU Baseline**, a **Naive GPU** implementation, and an **Optimized GPU** implementation, profiled on an NVIDIA GeForce RTX 3050 using NVIDIA Nsight Compute.

---

## Overview

This project takes a white-box approach. By rebuilding the same LeNet-5 forward pass three times from scratch and extracting granular hardware metrics at each stage, it establishes exact cause-and-effect relationships between algorithmic design decisions and the hardware bottlenecks they create or resolve.

The network was trained on CIFAR-10 in Python, and its weights were extracted into binary files. All three C++/CUDA implementations load these same weights and run inference on a single 32×32 RGB test image, producing identical predictions. Only performance changes.

---

## Architecture

LeNet-5 adapted for CIFAR-10 (3-channel RGB input, 10 output classes):

```
Input (3×32×32)
    → Conv1 (6 filters, 5×5) → 6×28×28
    → MaxPool1 (2×2)          → 6×14×14
    → Conv2 (16 filters, 5×5) → 16×10×10
    → MaxPool2 (2×2)          → 16×5×5
    → FC1 (400 → 120, ReLU)
    → FC2 (120 → 84, ReLU)
    → FC3 (84 → 10)
    → Prediction
```

---

## Hardware & Profiling Environment

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Architecture | Ampere (Compute Capability 8.6) |
| VRAM | 3 GB |
| SM Count | 16 |
| Warp Size | 32 |
| Max Threads/Block | 1024 |
| Max Threads/SM | 1536 |
| Profiler | NVIDIA Nsight Compute (`ncu`) |

---

## Implementations

### CPU Baseline (`infer_baseline.cpp`)

A direct algorithmic translation of each layer's math into standard C++. Seven nested `for` loops per convolution layer iterate over batch, output channels, output height, output width, input channels, kernel height, and kernel width. Serves as both the functional verification reference and performance baseline.

Key characteristic: the CPU's hardware prefetcher and L2/L3 caches automatically manage the overlapping 5×5 sliding windows, so the primary bottleneck is not memory but the absence of parallel ALUs.

### Naive GPU (`infer.cu`)

A structural translation of the CPU baseline into CUDA kernels following a strict **one thread per output element** paradigm.

- **Convolution layers** use a 3D grid: `dim3(16,16,1)` thread blocks, with grid dimensions computed via ceiling division `(OutW+15)/16 × (OutH+15)/16 × OutC`. Each thread independently computes its output pixel's full dot product from global memory.
- **FC layers** launch a single block of 128 threads. Each thread computes one output neuron by iterating over all input features sequentially. With only 1 block dispatched, only 1 of 16 SMs is ever active.

No shared or constant memory uses.

### Optimized GPU (`infer_optimized.cu`)

The naive implementation refactored to explicitly exploit the GPU's memory hierarchy through three targeted techniques:

**1. Shared Memory Tiling (Conv1, Conv2)**

Before computing any output, all threads in a block cooperatively load a 20×20 input tile into `__shared__` memory. After a `__syncthreads()` barrier, the convolution reads exclusively from shared memory to eliminate redundant DRAM fetches for overlapping receptive fields. For Conv2, the entire 6×14×14 input feature map (~4.7 KB) is loaded once and reused across all 16 output channels.

**2. Parallel Tree Reduction (FC1, FC2, FC3)**

Instead of mapping one thread to one output neuron, each output neuron gets its own block. Threads within a block cooperatively stride through the input vector, accumulate partial sums into `extern __shared__ float sdata[]`, then collapse them via a log₂-depth tree reduction. FC1's grid expands from 1 block → 120 blocks, distributing work across all 16 SMs.

**3. Constant Memory Broadcasting (Conv1, Conv2, FC2, FC3)**

Read-only filter weights are placed in `__constant__` memory via `cudaMemcpyToSymbol`. The constant cache broadcasts a single value to all 32 threads in a warp simultaneously, eliminating 32 separate DRAM transactions per filter coefficient access. The FC1 weight matrix (120×400 = 192 KB) exceeds the 64 KB constant memory limit and is retained in global memory.

---

## Results

### Execution Time

```
Layer    CPU (µs)    Naive GPU (µs)    Optimized GPU (µs)
------   --------    --------------    ------------------
Conv1      3,525           9.28                7.26
Pool1         56           2.75                3.01
Conv2      2,285          16.51               12.35
Pool2         16           2.69                2.94
FC1          164          25.54                5.02
FC2           28           7.52                6.05
FC3            2           6.24                4.29
------   --------    --------------    ------------------
Total      6,076         ~70.53              ~40.92
```

Total speedup: **CPU → Naive GPU ≈ 86×** | **CPU → Optimized GPU ≈ 148×** | **Naive → Optimized ≈ 1.72×**

### CPU → Naive GPU Speedup

| Layer | Speedup |
|-------|:-------:|
| Conv1 | **379.85×** |
| Pool1 | 20.36× |
| Conv2 | **138.40×** |
| Pool2 | 5.95× |
| FC1 | 6.42× |
| FC2 | 3.72× |
| FC3 | **0.32×** ⚠️ |

Conv layers dominate because high arithmetic intensity allows the GPU's ALUs to mask unoptimized memory latency. FC3's **0.32×** regression is caused by kernel launch overhead exceeding the cost of sequentially computing just 10 dot products on the host.

### Achieved Occupancy (Naive vs. Optimized)

| Kernel | Naive | Optimized | Δ |
|--------|:-----:|:---------:|:-:|
| conv1 | 20.65% | 23.39% | +2.74 pp |
| pool1 | 14.84% | 14.90% | ≈ flat |
| conv2 | 9.48% | 12.00% | +2.52 pp |
| pool2 | 10.32% | 10.09% | ≈ flat |
| **fc1** | **8.25%** | **55.36%** | **+47.11 pp** |
| fc2 | 6.26% | 19.66% | +13.40 pp |
| fc3 | 2.08% | 2.08% | flat |

The FC1 jump from 8.25% → 55.36% is the single most significant hardware improvement in the project, driven entirely by expanding the grid from 1 block to 120 blocks.

### Waves Per SM (Naive vs. Optimized)

| Kernel | Naive | Optimized |
|--------|:-----:|:---------:|
| conv1 | 0.25 | 0.25 |
| pool1 | 0.06 | 0.06 |
| fc1 | 0.01 | **0.62** |
| fc2 | 0.01 | **0.33** |
| fc3 | 0.00 | 0.04 |

Values below 1.0 indicate the GPU cannot fill a full wave; some SMs sit idle. Naive FC1 and FC2 at 0.01 means 15 of 16 SMs were completely inactive during those kernels.

### Throughput Diagnostics (Naive vs. Optimized)

| Layer | Naive DRAM% | Opt DRAM% | Naive L1% | Opt L1% | Naive Compute% | Opt Compute% |
|-------|:-----------:|:---------:|:---------:|:-------:|:--------------:|:------------:|
| Conv1 | 2.38 | 3.73 ↑ | 28.66 | 24.18 ↓ | 23.61 | 18.16 ↓ |
| Conv2 | 0.64 | 0.90 ↑ | 13.26 | 10.65 ↓ | 12.49 | 14.97 ↑ |
| FC1 | 4.08 | 20.80 ↑ | **83.18** | 34.47 ↓ | **1.02** | **24.74** ↑ |
| FC2 | 3.26 | 3.86 ↑ | 50.94 | 7.34 ↓ | 0.86 | **21.61** ↑ |

**Reading the throughput table:**
- Conv1's L1 drop (28.66% → 24.18%) is *correct*. Shared memory tiling replaced redundant L1 accesses with a single cooperative load, so less L1 traffic means the optimization is working.
- FC1's pairing of **83% L1 + 1% Compute** in the naive kernel is the hardware signature of serialization: one SM hammering the cache with sequential requests while 15 SMs and all compute units sit idle.
- After optimization, FC1's compute jumps to **24.74%** as the parallel reduction spreads real arithmetic work across all SMs.

---

## Key Findings

**1. Parallelism alone is not optimization.**
The Naive GPU achieved massive speedups for convolutional layers (up to 380×) purely through thread parallelism. But for FC layers, a structurally broken grid configuration (1 block for 120 neurons) capped the speedup at 6× and caused FC3 to regress below CPU performance. Raw thread count without proper grid design wastes the hardware.

**2. The 1D grid is a structural flaw, not a tuning issue.**
Naive FC layers didn't need better block sizes or more threads per block. They needed a fundamentally different parallelization strategy. Parallel tree reduction changed the question from "how many threads compute one neuron?" to "how many neurons run in parallel?", which is what enabled the 5× additional speedup on FC1.

**3. Lower memory traffic is a sign of optimization, not inefficiency.**
The L1 throughput drops seen in Conv1 after shared memory tiling are frequently misread as degraded performance. They are the opposite; they confirm that redundant DRAM reads were successfully eliminated. The meaningful metric to watch is execution time and compute throughput, both of which improve.

**4. LeNet-5 on a single image is too small for modern GPU hardware.**
Even in its fully optimized state, the peak hardware saturation reached only **0.62 waves per SM**. The RTX 3050 has far more physical execution capacity than a single 32×32 RGB image can generate. This is not a code problem, but rather a workload size problem. The most impactful path forward is **batched inference**: running N images simultaneously scales every grid proportionally and is the only way to push waves per SM past 1.0 on this architecture.

---

## Project Structure

```
.
├── include/
│   └── utils/
│       └── file_io.h            # Shared binary file I/O utilities
├── src/
│   ├── baseline/
│   │   └── infer_baseline.cpp   # Sequential CPU implementation
│   └── cuda/
│       ├── infer.cu             # Naive GPU — full forward pass
│       ├── infer_optimized.cu   # Optimized GPU — full forward pass
│       ├── conv1.cu             # Naive conv1 kernel
│       ├── conv2.cu             # Naive conv2 kernel
│       ├── pool1.cu             # Naive pool1 kernel
│       ├── pool2.cu             # Naive pool2 kernel
│       ├── fc1.cu               # Naive fc1 kernel
│       ├── fc2.cu               # Naive fc2 kernel
│       └── fc3.cu               # Naive fc3 kernel
└── weights/                     # Extracted PyTorch model weights (.bin)
    ├── conv1_weight.bin
    ├── conv1_bias.bin
    ├── conv2_weight.bin
    ├── conv2_bias.bin
    ├── fc1_weight.bin
    ├── fc1_bias.bin
    ├── fc2_weight.bin
    ├── fc2_bias.bin
    ├── fc3_weight.bin
    └── fc3_bias.bin
```

---

## Building and Running

**CPU Baseline**
```bash
g++ -O2 -I include -o infer_baseline src/baseline/infer_baseline.cpp
./infer_baseline
```

**Naive GPU**
```bash
nvcc -O2 -I include -o infer src/cuda/infer.cu
./infer
```

**Optimized GPU**
```bash
nvcc -O2 -I include -o infer_optimized src/cuda/infer_optimized.cu
./infer_optimized
```

**Profiling with Nsight Compute**
```bash
ncu --set full -o profile_naive ./infer
ncu --set full -o profile_optimized ./infer_optimized
```

---

## Technologies

- **C++ / CUDA C++**: all three implementations
- **NVIDIA Nsight Compute**: hardware profiling
- **Python / PyTorch**: model training and weight extraction
- **NVIDIA GeForce RTX 3050**: Ampere, CC 8.6, 16 SMs

---

*University of the Philippines Diliman — Electrical and Electronics Engineering Institute*