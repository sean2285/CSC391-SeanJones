import cv2
import matplotlib.pyplot as plt
from contrast_stretch import contrast_stretch
from equalize_histogram import equalize_histogram
from calculate_histogram import calculate_histogram

# === Load low-contrast image ===
img = cv2.imread("images/low_contrast.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Make sure it's in the 'images/' folder.")

# === Apply transformations ===
stretched_img = contrast_stretch(img, r_min=50, r_max=180)
equalized_img = equalize_histogram(img)

# === Compute histograms ===
counts_original, _ = calculate_histogram(img)
counts_stretched, _ = calculate_histogram(stretched_img)
counts_equalized, _ = calculate_histogram(equalized_img)

# === Plot histograms side-by-side ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(range(256), counts_original, width=1)
plt.title("Original Histogram")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")

plt.subplot(1, 3, 2)
plt.bar(range(256), counts_stretched, width=1)
plt.title("Contrast Stretched Histogram")
plt.xlabel("Intensity Value")

plt.subplot(1, 3, 3)
plt.bar(range(256), counts_equalized, width=1)
plt.title("Equalized Histogram")
plt.xlabel("Intensity Value")

plt.tight_layout()
plt.show()
