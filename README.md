# CUDA Runtime Playground

This project is a playground for experimenting with CUDA, focusing on finding the highest sum pairs by column and row in matrices. The goal is to compare GPU vs CPU performance and explore scenarios where CPU might outperform GPU for smaller matrices. But mostly a way for me to play with CUDA!

## Description

The code implements algorithms to compute the highest sum pairs across rows and columns of matrices using both CUDA (GPU) and CPU implementations. It includes timing measurements to analyze performance differences.

I have started with the naive implementation of the algorithm; I intend to add more optimised variations soon (doing a matrix rotation, and doing some tiled rotations).

This line lets me loop over multiple matrix sizes:
```
    // sample list of matric sizes to test; smaller ones can be faster on CPU
    std::vector<int> sizes = { 8, 1024, 5000, 10000 };
```

Key observations:
- GPU acceleration shines for larger matrices
- For smaller matrices, CPU can sometimes be faster due to overhead of GPU kernel launches and data transfers

Sample output -
```
========================================
Matrix size: 5000 x 5000

Timing (ms)
Operation       CPU (ms)        GPU (ms)        Speedup
----------------------------------------------------------------
Row             102.381         2.293           44.653          x
Column          244.582         3.180           76.905          x

Transfer:
  H2D: 30.003 ms
  D2H: 0.094 ms

========================================
```


## Requirements

- CUDA Toolkit (compatible with your GPU)
- Visual Studio (for building the .vcxproj)
- NVIDIA GPU with CUDA support

## Building

1. Open `CudaRuntime.sln` in Visual Studio.
2. Build the solution
3. The executable will be generated in `x64/Debug/` or `x64/Release/`.

## Running

Execute the built `CudaRuntime.exe` from the command line or debugger. The program will run benchmarks comparing CPU and GPU implementations.

## Files

- `kernel.cu`: CUDA kernel and host code

## Notes

This is experimental code for learning CUDA performance characteristics. Results may vary based on hardware, matrix sizes, and CUDA version.  I built against a RTX5060.