import numpy as np
from numba import cuda
from numba import float32
from scipy.linalg import blas
import time

blas_lib = blas

# Tile size
tile_size = 32

@cuda.jit
def tiled_matmul(A, B, C):
    # Define shared memory for tiles
    sA = cuda.shared.array(shape=(tile_size, tile_size), dtype=float32)
    sB = cuda.shared.array(shape=(tile_size, tile_size), dtype=float32)

    # Define thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # Define row and column indices
    row = by * bh + ty
    col = bx * bw + tx

    # Initialize C[row, col]
    c_val = 0.0

    # Loop over tiles
    for i in range(0, A.shape[1], tile_size):
        # Load tiles into shared memory
        if row < A.shape[0] and i + tx < A.shape[1]:
            sA[ty, tx] = A[row, i + tx]
        else:
            sA[ty, tx] = 0.0

        if i + ty < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[i + ty, col]
        else:
            sB[ty, tx] = 0.0

        # Synchronize threads
        cuda.syncthreads()

        # Compute tile multiplication
        for j in range(tile_size):
            c_val += sA[ty, j] * sB[j, tx]

        # Synchronize threads
        cuda.syncthreads()

    # Write result to C
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = c_val

def measure_execution_time_tiled(N):
    # Configuration
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C_kernel = np.zeros((N, N), dtype=np.float32)
    C_cblas = np.zeros((N, N), dtype=np.float32)

    # Copy the arrays to the device
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C_kernel = cuda.to_device(C_kernel)

    # Configure the blocks
    threadsperblock = (tile_size, tile_size)
    blockspergrid_x = (N + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (N + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Create events for timing
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record()

    # Launch the kernel
    tiled_matmul[blockspergrid, threadsperblock](d_A, d_B, d_C_kernel)

    # Record end event
    end.record()
    end.synchronize()
  
    # Calculate the execution time
    execution_time = cuda.event_elapsed_time(start, end)

    # Copy the result back to the host
    d_C_kernel.copy_to_host(C_kernel)

    return C_kernel, execution_time

# Number of runs for averaging
num_runs = 100

# Array of sizes for the sweep
sizes = [2**i for i in range(6, 14)]  # From 64 to 8192

# Dictionary to store average execution times and errors for each size
average_times_tiled = {}
average_errors_tiled = {}

for N in sizes:
    total_time = 0
    total_error = 0
    for _ in range(num_runs):
        # Re-initialize matrices A and B for each size N
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C_kernel, execution_time = measure_execution_time_tiled(N)

        # Call cblas_sgemm
        start = time.time()
        C_cblas = blas.sgemm(1.0, A, B, 0.0, np.zeros((N, N), dtype=np.float32))
        cblas_time = time.time() - start

        total_time += execution_time
        total_error += np.linalg.norm(C_kernel - C_cblas) / np.linalg.norm(C_cblas)  # Relative error

    average_time = total_time / num_runs
    average_error = total_error / num_runs
    average_times_tiled[N] = average_time
    average_errors_tiled[N] = average_error
    print(f"Average execution time for N={N}: {average_time} ms")
    print(f"Average error for N={N}: {average_error}")