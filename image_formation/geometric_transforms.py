import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
image_path = "images/original_image.jpg"  # update this path
original_image = cv2.imread(image_path)

# Check image load
if original_image is None:
    raise FileNotFoundError("Image not found. Check path!")

# Convert BGR to RGB for displaying with Matplotlib
original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# --- Perspective Transformation ---

rows, cols = original_image.shape[:2]

# 1. Source points (corners of the original image)
pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])

# 2. Destination points (adjust numbers to control skew/tilt)
pts2 = np.float32([
    [100, 50],          # top-left corner
    [cols-200, 0],      # top-right corner
    [50, rows-50],      # bottom-left corner
    [cols-100, rows-100]  # bottom-right corner
])

# 3. Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# 4. Apply warpPerspective
transformed_image = cv2.warpPerspective(original_image, M, (cols, rows))

# Convert to RGB for display
transformed_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

# --- Reverse Engineering the Transform ---
M_inv = np.linalg.inv(M)  # inverse matrix
reverse_engineered_image = cv2.warpPerspective(transformed_image, M_inv, (cols, rows))
reverse_rgb = cv2.cvtColor(reverse_engineered_image, cv2.COLOR_BGR2RGB)

# --- Display ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(transformed_rgb)
plt.title("Transformed Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reverse_rgb)
plt.title("Reverse-Engineered Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- Print applied transforms ---
print("Transformations applied:")
print(f"Perspective warp with points:\nSource: {pts1}\nDestination: {pts2}")
print("Reverse transformation applied using inverse matrix.")