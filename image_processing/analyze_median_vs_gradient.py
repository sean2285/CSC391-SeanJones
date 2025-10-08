import cv2
import numpy as np
import matplotlib.pyplot as plt
from median_filter import median_filter
from calculate_gradient import calculate_gradient

# Load grayscale image
img = cv2.imread("images/low_contrast.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found in images/ folder")

# Add salt-and-pepper noise
def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    num_pixels = int(amount * image.size)
    # Salt (white pixels)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

noisy_img = add_salt_pepper_noise(img)

# Apply median filter
filtered_img = median_filter(noisy_img, size=3)

# Compute gradient magnitudes
grad_noisy = calculate_gradient(noisy_img)
grad_filtered = calculate_gradient(filtered_img)

# Display comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title("Noisy Image")

plt.subplot(2, 2, 2)
plt.imshow(filtered_img, cmap='gray')
plt.title("After Median Filtering")

plt.subplot(2, 2, 3)
plt.imshow(grad_noisy, cmap='gray')
plt.title("Gradient Magnitude (Noisy Image)")

plt.subplot(2, 2, 4)
plt.imshow(grad_filtered, cmap='gray')
plt.title("Gradient Magnitude (After Median Filtering)")

plt.tight_layout()
plt.show()
