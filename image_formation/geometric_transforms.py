import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images/original_image.jpg"
original_image = cv2.imread(image_path)

if original_image is None:
    raise FileNotFoundError("Image not found. Check path!")

original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
rows, cols = original_image.shape[:2]

pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
pts2 = np.float32([
    [cols//3, 0],
    [cols//2, 50],
    [0, rows],
    [cols, rows]
])

M = cv2.getPerspectiveTransform(pts1, pts2)
transformed_image = cv2.warpPerspective(original_image, M, (cols*2, rows*2))
transformed_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(transformed_rgb)
plt.title("Transformed Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Transformations applied:")
print(f"Source points:\n{pts1}")
print(f"Destination points:\n{pts2}")

