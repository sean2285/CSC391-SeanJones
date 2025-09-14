import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images/original_image.jpg"
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError("Image not found. Check path!")

original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
rows, cols = original_image.shape[:2]

src_points = np.float32([
    [0, 0],
    [cols - 1, 0],
    [cols - 1, rows - 1],
    [0, rows - 1]
])

dst_points = np.float32([
    [cols * 0.2, 0],
    [cols * 0.9, rows * 0.1],
    [cols * 0.8, rows * 1.2],
    [cols * 0.5, rows * 1.2]
])

M = cv2.getPerspectiveTransform(src_points, dst_points)
warped = cv2.warpPerspective(original_image, M, (cols*2, rows*2))
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(warped_rgb)
plt.title("Transformed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
