import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# Define a CUDA kernel that performs arithmetic operations
kernel_code = """
__global__ void compute(float *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        for (int i = 0; i < 1000; i++) {  // Loop to keep cores busy
            data[idx] = (data[idx] * 2.0f) - (data[idx] / 3.0f) + 1.5f;
        }
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)
compute = mod.get_function("compute")

# Number of elements in the array (adjust for GPU capacity)
N = 1000000
data = np.random.rand(N).astype(np.float32)

# Allocate GPU memory
data_gpu = cuda.mem_alloc(data.nbytes)

# Copy data to GPU
cuda.memcpy_htod(data_gpu, data)

# Define grid and block sizes
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Run the kernel actively for 1 minute
start_time = time.time()
while time.time() - start_time < 60:
    compute(data_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy the result back to the CPU (optional)
cuda.memcpy_dtoh(data, data_gpu)

# Free GPU memory
data_gpu.free()

print("GPU computation completed for 1 minute.")
