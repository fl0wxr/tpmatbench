import time
import os
import argparse
import re


def available_ram() -> int:
  """
  Description:
    Compute and get available system RAM.

  Returns:
    `bytes_` -- Type: int. Available system RAM in bytes.
  """

  with open("/proc/meminfo", "r") as f:
    meminfo = f.read()

  for line in meminfo.split("\n")[:-1]:

    if re.match(
      pattern = r"^MemAvailable",
      string = line
    ):

      bytes_ = int(line.split()[1]) * 1024

      return bytes_

def available_vram() -> int:
  """
  Description:
    Compute and get available vRAM. Depends on an already deployed NVIDIA Toolkit.

  Returns:
    `bytes_` -- Type: int. Available vRAM in bytes.
  """

  bytes_ = 0
  n_gpu = cp.cuda.runtime.getDeviceCount()

  for gpu_idx in range(int(n_gpu)):
    bytes_ += cp.cuda.Device(gpu_idx).mem_info[0]

  return bytes_

def operand_size(precision: int, free_mem: int, acceleration_factor: float = 1) -> int:
  """
  Description:
    Compute the operands' sizes based on free memory.

  Parameter:
    `precision` -- Type int. Domain in {32, 64}. Precision of floating point numbers of benchmark operands in bits.
    `free_mem` -- Type int. Available memory in bytes.
    `acceleration_factor` -- Type float. Higher values lead to lower n for a lower amount of benchmark computations (relative to available/free memory). Conversely, lower values lead to a higher amount of benchmark computations. Default parameter is set to 1.

  Returns:
    `n` -- Type int. The sizes of the square operand matrices.
  """

  benchmark_factor = 0.8  # Fraction of memory that will be used by default; the rest is used as a margin for the system's dynamic overhead.
  usable_bytes = free_mem * benchmark_factor

  precision_bytes = precision//8

  # 3 -> 3 matrices (operand plus output); `precision_bytes` bytes per operand element.
  n = int((usable_bytes / (3 * precision_bytes)) ** 0.5 / acceleration_factor)
  
  return n

def benchmark_cpu(n: int, precision: int):
  """
  Description:
    Measure CPU FLOPs throughput using (dense) matrix multiplication.

  Parameter:
    `n` -- Type int. Operands' sizes.
    `precision` -- Type int. Domain in {32, 64}. Precision of floating point numbers of benchmark operands in bits.
  """

  if precision == 32:
    dtype = np.float32
  else:
    dtype = np.float64

  print(f"CPU Benchmark\nOperation scale: {n}\nOperands precision: {precision} bits")

  A = np.random.rand(n, n).astype(dtype)
  B = np.random.rand(n, n).astype(dtype)

  # Warm-up.
  _ = A @ B
  del _

  t_ini = time.time()
  C = A @ B
  t_fin = time.time()

  delta_t = t_fin - t_ini
  flops = 2 * n**3 / delta_t
  gflops = flops / 1e9

  print(f"Elapsed time: {delta_t:.3f} s\nEstimated GFLOPS: {gflops:.2f}")

def benchmark_gpu(n: int, precision: int):
  """
  Description:
    Measure GPU FLOPs throughput using (dense) matrix multiplication.

  Parameter:
    `n` -- Type int. Operands' sizes.
    `precision` -- Type int. Domain in {32, 64}. Precision of floating point numbers of benchmark operands in bits.
  """

  if precision == 32:
    dtype = cp.float32
  else:
    dtype = cp.float64

  print(f"GPU Benchmark\nOperation scale: {n}\nOperands precision: {dtype} bits")

  A = cp.random.rand(n, n, dtype=dtype)
  B = cp.random.rand(n, n, dtype=dtype)

  # Warm-up.
  _ = A @ B
  cp.cuda.Stream.null.synchronize()
  del _

  t_ini = time.time()
  C = A @ B
  cp.cuda.Stream.null.synchronize()
  t_fin = time.time()

  delta_t = t_fin - t_ini
  flops = 2 * n**3 / delta_t
  gflops = flops / 1e9

  print(f"Elapsed time: {delta_t:.3f} s\nEstimated GPU GFLOPS: {gflops:.2f}")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description="\
tpmatbench -- Matrix Algebra Throughput Benchmarking Utility.\n\
\n\
Description: Return FLOPs of current machine after stress testing it using a primitive FP32 matrix algebra operation.\n\
The end-user must ensure that all threads are utilized together for at least 1 minute for an accurate measurement. \
If it takes too long to complete, then interrupt it and use the --acceleration argument appropriately.\
    ",
    formatter_class=argparse.RawTextHelpFormatter  # Preserve newline characters in stdout of `--help`.
)
  parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help='Choose "cpu" for NumPy or "gpu" for CuPy. Default parameter is set to "cpu".')
  parser.add_argument("--precision", type=int, default=32, choices=[32, 64], help="Default Precision of floating point numbers of benchmark operands in bits. Default parameter is set to 32 .")
  parser.add_argument("--acceleration", type=float, default=1.0, help="Increase if the runtime takes an excessive amount of time (this will shorten the runtime); otherwise decrease. Default parameter is set to 1.0 .")
  args = parser.parse_args()

  assert parser.acceleration > 0, "E: Acceleration factor must be a positive number."

  if args.device == "gpu":

    import cupy as cp

    free_mem = available_vram(precision=args.precision)
    n = operand_size(free_mem=free_mem, acceleration_factor=args.acceleration)
    benchmark_gpu(n=n, precision=args.precision)

  else:

    import numpy as np

    free_mem = available_ram(precision=args.precision)
    n = operand_size(free_mem=free_mem, acceleration_factor=args.acceleration)
    benchmark_cpu(n=n, precision=args.precision)
