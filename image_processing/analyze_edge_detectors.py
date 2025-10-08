import cv2
import numpy as np
import matplotlib.pyplot as plt
from sobel_edge_detector import sobel_edge_detector
from directional_edge_detector import directional_edge_detector
from calculate_gradient import calculate_gradient

# Load grayscale test image
img = cv2.imread("images/low_contrast.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found in images/ folder")

# Apply Sobel edge detector (you can tune threshold = 80–120)
edge_sobel = sobel_edge_detector(img, threshold=100)

# Only keep edges where gradient direction ≈ 45°
edge_dir_45 = directional_edge_detector(img, direction_range=(40, 50))

# Automatically detects edges based on gradient and hysteresis thresholds
edge_canny = cv2.Canny(img, 100, 200)

# Display results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(edge_sobel, cmap="gray")
plt.title("Sobel Edge Map (Magnitude Only)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edge_dir_45, cmap="gray")
plt.title("Directional Edge Map (~45°)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edge_canny, cmap="gray")
plt.title("Canny Edge Map (OpenCV)")
plt.axis("off")

plt.tight_layout()
plt.show()
