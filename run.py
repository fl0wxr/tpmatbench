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

def operand_size(free_mem: int, acceleration_factor: float = 1) -> int:
  """
  Description:
    Compute the operands' sizes based on free memory.

  Parameters:
    `free_mem` -- Type int. Available memory in bytes.
    `acceleration_factor` -- Type float. Higher values lead to lower n for a lower amount of benchmark computations (relative to available/free memory). Conversely, lower values lead to a higher amount of benchmark computations. Its default parameter is set to 1, making the matrices 

  Returns:
    `n` -- Type int. The sizes of the square operand matrices.
  """

  assert acceleration_factor > 0, "E: Acceleration factor must be a positive number."

  benchmark_factor = 0.8  # Fraction of memory that will be used by default; the rest is used as a margin for the system's dynamic overhead.
  usable_bytes = free_mem * benchmark_factor

  # 3 -> 3 matrices (operand plus output); 4 bytes (float 32 bit) per operand element.
  n = int((usable_bytes / (3 * 4)) ** 0.5 / acceleration_factor)
  
  return n

def benchmark_cpu(n: int):
  """
  Description:
    Measure CPU FLOPs throughput using (dense) matrix multiplication.

  Parameter:
    `n` -- Type int. Operands' sizes.
  """

  print(f"CPU Benchmark\nOperation scale: {n}")

  A = np.random.rand(n, n).astype(np.float32)
  B = np.random.rand(n, n).astype(np.float32)

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

def benchmark_gpu(n: int):
  """
  Description:
    Measure GPU FLOPs throughput using (dense) matrix multiplication.

  Parameter:
    `n` -- Type int. Operands' sizes.
  """

  print(f"GPU Benchmark\nOperation scale: {n}")

  A = cp.random.rand(n, n, dtype=cp.float32)
  B = cp.random.rand(n, n, dtype=cp.float32)

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
Description: Return FLOPs of current machine after stress testing it using a primitive matrix algebra operation.\n\
The end-user must ensure that all threads are utilized together for at least 1 minute for an accurate measurement. \
If it takes too long to complete, then interrupt it and use the --acceleration argument appropriately.\
    ",
    formatter_class=argparse.RawTextHelpFormatter  # Preserve newline characters in stdout of --help.
)
  parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help='Choose "cpu" for NumPy or "gpu" for CuPy.')
  parser.add_argument("--acceleration", type=float, default=1.0, help="Increase if the runtime takes an excessive amount of time (this will shorten the runtime); otherwise decrease.")
  args = parser.parse_args()

  if args.device == "gpu":

    import cupy as cp

    free_mem = available_vram()
    n = operand_size(free_mem=free_mem, acceleration_factor=args.acceleration)
    benchmark_gpu(n=n)

  else:

    import numpy as np

    free_mem = available_ram()
    n = operand_size(free_mem=free_mem, acceleration_factor=args.acceleration)
    benchmark_cpu(n=n)
