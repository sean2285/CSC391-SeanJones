import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original image
image_path = "images/original_image.jpg" 
original_image = cv2.imread(image_path)

if original_image is None:
    raise FileNotFoundError("Image not found. Check path!")

original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
rows, cols = original_image.shape[:2]

# Define source points (corners of the original image)
src_points = np.float32([
    [0, 0],
    [cols - 1, 0],
    [cols - 1, rows - 1],
    [0, rows - 1]
])

# Define destination points (skewed quadrilateral)
# Adjust these numbers to control the "tilt"
dst_points = np.float32([
    [cols * 0.2, 0],         # top-left pushed inward
    [cols * 0.9, rows * 0.1],# top-right pushed inward and slightly down
    [cols * 0.8, rows * 1.2],# bottom-right pushed inward and down
    [cols * 0.1, rows * 0.9] # bottom-left pushed inward
])

# Perspective transform matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply warp
warped = cv2.warpPerspective(original_image, M, (cols*2, rows*2))
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

# Show results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(warped_rgb)
plt.title("Leaning Card Effect")
plt.axis("off")

plt.tight_layout()
plt.show()

