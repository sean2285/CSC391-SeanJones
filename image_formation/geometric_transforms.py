import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images/original_image.jpg"  # update this path
original_image = cv2.imread(image_path)

if original_image is None:
    raise FileNotFoundError("Image not found. Check path!")

original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
rows, cols = original_image.shape[:2]

# 1. Rotation around image center
center = (cols // 2, rows // 2)
angle = -20   # adjust rotation angle (negative = clockwise)
scale = 1.0   # keep original scale
R = cv2.getRotationMatrix2D(center, angle, scale)

# 2. Add translation (shift the rotated board)
tx, ty = 100, 50   # adjust translation in x and y
R[:,2] += [tx, ty]

# 3. Apply affine transformation
rotated_translated = cv2.warpAffine(original_image, R, (cols*2, rows*2))
rotated_rgb = cv2.cvtColor(rotated_translated, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(rotated_rgb)
plt.title("Transformed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
