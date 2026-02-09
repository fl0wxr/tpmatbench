# TPMATBench: Throughput Matrix Algebra Benchmarking Utility

## Objective

TPMATBench is a Python benchmark that measures CPU and GPU 64-bit floating-point throughput (GFLOPS) using dense matrix multiplication. It is designed to stress-test a digital computer with a primitive matrix algebra operation.

## Deployment

Simply `git clone` this project.

## Usage

Navigate inside the project directory and understand the simple instructions given in `python3 run.py --help` for proper usage.

## Features

- Measures approximate GFLOPS for square matrix multiplication.
- Supports CPU (NumPy) and GPU (CuPy) if available.
- Configurable matrix size, and acceleration factor.

## Requirements

- Python 3
- NumPy
- Optional for GPU: CuPy (for CUDA-enabled GPUs)
