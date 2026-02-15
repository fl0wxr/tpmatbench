# TPMATBench: Throughput Matrix Algebra Benchmarking Utility

## Objective

TPMATBench is a Python benchmark that measures CPU and GPU throughput (GFLOPS) using dense FP32 matrix multiplication. It is designed to stress-test a digital computer with a primitive matrix algebra operation.

## Deployment

Simply `git clone` this project.

## Usage

Navigate inside the project directory and understand the simple instructions given in `python3 run.py --help` for proper usage. The end-user must ensure that all threads are utilized together for at least 1 minute for an accurate measurement.

Warning: It is advised to shut down any non-essential process that uses resources (volatile memory and processing power).

## Features

- Measures approximate GFLOPS for square matrix multiplication.
- Supports CPU (NumPy) and GPU (CuPy) if available.
- Configurable acceleration parameter that controls the operand matrix size.
- Configurable floating point precision of the operand matrix element.

## Requirements

- Python 3
- NumPy
- Optional for GPU: CuPy (for CUDA-enabled GPUs)
