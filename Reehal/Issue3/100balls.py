import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Set up the display
screen_width = 800  # Width of the screen
screen_height = 600  # Height of the screen
screen = pygame.display.set_mode((screen_width, screen_height))  # Create the display window
pygame.display.set_caption("Attractive Graphics Initialization")  # Set window title

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
colors = [BLUE, RED, GREEN, YELLOW, PURPLE]

# Load background image (if you have one, replace this path with your image)
# If you don't have a background image, you can skip this part and use a solid color.
# background_image = pygame.image.load("background.jpg")
# background_image = pygame.transform.scale(background_image, (screen_width, screen_height))

# Font for rendering text
font = pygame.font.SysFont("Arial", 36)

# Initial position for the moving circle
circle_x = 100
circle_y = 100
circle_radius = 30
circle_speed_x = 5
circle_speed_y = 5

# Main loop to keep the window open
running = True
while running:
    # Check for events (like closing the window)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with a gradient color effect (optional)
    screen.fill(random.choice(colors))  # Random color background

    # Blit the background image
    # screen.blit(background_image, (0, 0))

    # Display text in the middle of the screen
    text = font.render("Hello, Pygame!", True, WHITE)
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2 - 100))
    screen.blit(text, text_rect)

    # Draw a moving circle that bounces off the walls
    pygame.draw.circle(screen, RED, (circle_x, circle_y), circle_radius)

    # Update the position of the circle
    circle_x += circle_speed_x
    circle_y += circle_speed_y

    # Bounce the circle off the screen edges
    if circle_x + circle_radius >= screen_width or circle_x - circle_radius <= 0:
        circle_speed_x = -circle_speed_x
    if circle_y + circle_radius >= screen_height or circle_y - circle_radius <= 0:
        circle_speed_y = -circle_speed_y

    # Update the display
    pygame.display.flip()

    # Set the frame rate (FPS)
    pygame.time.Clock().tick(60)

# Clean up and close pygame
pygame.quit()
sys.exit()
