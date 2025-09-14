import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
image_path = "images/original_image.jpg"  # update this path
original_image = cv2.imread(image_path)

if original_image is None:
    raise FileNotFoundError("Image not found. Check path!")

# Convert BGR → RGB for matplotlib
original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

rows, cols = original_image.shape[:2]

# --- Perspective Transformation ---

# 1. Source points: corners of original image
pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])

# 2. Destination points: skew to create the "lean back" effect
pts2 = np.float32([
    [100, 0],          # top-left (pulled a bit right)
    [cols-200, 100],   # top-right (pushed down)
    [0, rows-1],       # bottom-left stays in place
    [cols-300, rows-1] # bottom-right pulled left
])

# 3. Compute perspective transform
M = cv2.getPerspectiveTransform(pts1, pts2)

# 4. Warp perspective - increase output size for full view
transformed_image = cv2.warpPerspective(original_image, M, (cols*2, rows*2))

# Convert to RGB for display
transformed_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

# --- Display ---
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(transformed_rgb)
plt.title("Transformed Image (Desired Look)")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Transformations applied:")
print(f"Source points:\n{pts1}")
print(f"Destination points:\n{pts2}")
