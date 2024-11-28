import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np

# Example CUDA kernel
mod = SourceModule("""
__global__ void square(float *a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    a[idx] *= a[idx];
}
""")
square = mod.get_function("square")
print (square)

# Run the kernel

# Input array
n = 10  # Number of elements
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)

# Copy input array to GPU
cuda.memcpy_htod(a_gpu, a)

# Launch the kernel (1 block with n threads)
threads_per_block = 10
blocks_per_grid = 1
square(a_gpu, block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))

# Copy the result back to the CPU
result = np.empty_like(a)
cuda.memcpy_dtoh(result, a_gpu)

# Print the result
print("Original array:", a)
print("Squared array:", result)