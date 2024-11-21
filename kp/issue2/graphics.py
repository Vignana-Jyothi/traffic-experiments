import pyopencl as cl
import pygame
import numpy as np
import time

# Initialize OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OpenCL kernel code
kernel_code = """
__kernel void update_positions(
    __global float *positions,
    __global float *velocities,
    const float width,
    const float height,
    const float dt) {
    int idx = get_global_id(0);
    int offset = idx * 2; // Each ball has x, y coordinates

    // Update positions
    positions[offset] += velocities[offset] * dt;     // Update x
    positions[offset + 1] += velocities[offset + 1] * dt; // Update y

    // Check for collisions with walls
    if (positions[offset] <= 0 || positions[offset] >= width) {
        velocities[offset] = -velocities[offset];
    }
    if (positions[offset + 1] <= 0 || positions[offset + 1] >= height) {
        velocities[offset + 1] = -velocities[offset + 1];
    }
}
"""

# Compile OpenCL program
program = cl.Program(context, kernel_code).build()

# Initialize canvas
canvas_width, canvas_height = 800, 600
pygame.init()
screen = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("Moving Balls with OpenCL")
clock = pygame.time.Clock()

# Ball data
num_balls = 2
# Initialize positions with separate x and y values
positions = np.zeros(num_balls * 2, dtype=np.float32)
positions[::2] = np.random.rand(num_balls).astype(np.float32) * canvas_width  # x positions
positions[1::2] = np.random.rand(num_balls).astype(np.float32) * canvas_height  # y positions


velocities = (np.random.rand(num_balls * 2).astype(np.float32) - 0.5) * 40000  # Random velocities

# Create OpenCL buffers
positions_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=positions)
velocities_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=velocities)

# Simulation parameters
dt = 0.001  # Time step

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Run OpenCL kernel to update ball positions
    program.update_positions(
        queue,
        (num_balls,),
        None,
        positions_buf,
        velocities_buf,
        np.float32(canvas_width),
        np.float32(canvas_height),
        np.float32(dt)
    )

    # Read updated positions from the GPU
    cl.enqueue_copy(queue, positions, positions_buf)

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw balls
    for i in range(num_balls):
        x, y = int(positions[i * 2]), int(positions[i * 2 + 1])
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 20)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

# Cleanup
pygame.quit()
