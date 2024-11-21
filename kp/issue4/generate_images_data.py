import cv2
import numpy as np
import os

def create_shapes(output_dir, num_images):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_images):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        shape = np.random.choice(['rectangle', 'triangle'])
        if shape == 'rectangle':
            cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        else:  # triangle
            points = np.array([[50, 10], [10, 80], [90, 80]], np.int32)
            cv2.fillPoly(img, [points], (255, 255, 255))
        cv2.imwrite(f"{output_dir}/{shape}_{i}.png", img)

create_shapes('shapes_dataset', 1000)
