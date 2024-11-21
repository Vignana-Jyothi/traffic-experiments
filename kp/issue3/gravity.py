import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Physics Simulator")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Clock
clock = pygame.time.Clock()
FPS = 60

# Particle class
class Particle:
    def __init__(self, x, y, radius, color, mass):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
    
    def move(self):
        # Update position
        self.x += self.vx
        self.y += self.vy

        # Apply gravity
        self.vy += 0.1  # Gravity force
    
        # Collision with walls
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.vx *= -1
        if self.y - self.radius < 0 or self.y + self.radius > HEIGHT:
            self.vy *= -1

    def collide(self, other):
        # Check for collision with another particle
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance < self.radius + other.radius:
            # Elastic collision response
            nx = dx / distance
            ny = dy / distance
            p = 2 * (self.vx * nx + self.vy * ny - other.vx * nx - other.vy * ny) / (self.mass + other.mass)
            self.vx -= p * self.mass * nx
            self.vy -= p * self.mass * ny
            other.vx += p * other.mass * nx
            other.vy += p * other.mass * ny

# Generate particles
particles = [
    Particle(
        random.randint(50, WIDTH - 50),
        random.randint(50, HEIGHT - 50),
        random.randint(10, 20),
        random.choice([RED, BLUE, BLACK, GREEN]),
        random.randint(1, 5)
    ) for _ in range(20)
]

# Main loop
running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update particles
    for i, particle in enumerate(particles):
        particle.move()
        for other in particles[i + 1:]:
            particle.collide(other)
        particle.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
