import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
image_path = "images/original_image.jpg"  # Replace with your image path
original_image = cv2.imread(image_path)

# Convert BGR to RGB for displaying with Matplotlib
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Apply a 2D geometric transformation (e.g., rotation)
# Define the rotation matrix
angle = 45  # Rotation angle in degrees
center = (original_image.shape[1] // 2, original_image.shape[0] // 2)  # Center of the image
scale = 1.0  # Scaling factor
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation
transformed_image = cv2.warpAffine(original_image, rotation_matrix, (original_image.shape[1], original_image.shape[0]))

# Convert transformed image to RGB
transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

# Reverse-engineer the transformation (rotate back)
reverse_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
reverse_engineered_image = cv2.warpAffine(transformed_image, reverse_rotation_matrix, (original_image.shape[1], original_image.shape[0]))

# Convert reverse-engineered image to RGB
reverse_engineered_image_rgb = cv2.cvtColor(reverse_engineered_image, cv2.COLOR_BGR2RGB)

# Display the images side by side
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(original_image_rgb)
plt.title("Original Image")
plt.axis("off")

# Transformed image
plt.subplot(1, 3, 2)
plt.imshow(transformed_image_rgb)
plt.title("Transformed Image")
plt.axis("off")

# Reverse-engineered image
plt.subplot(1, 3, 3)
plt.imshow(reverse_engineered_image_rgb)
plt.title("Reverse-Engineered Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print the transformations applied
print("Transformations applied:")
print(f"1. Rotation: {angle} degrees")
print("2. Reverse Rotation: -{angle} degrees")