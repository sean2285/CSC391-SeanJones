import cv2
import numpy as np

def median_filter(img, size=3):
    
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Get padding for edges
    pad = size // 2
    padded_img = np.pad(img, pad, mode='edge')
    filtered_img = np.zeros_like(img)

    # Apply median filter manually
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded_img[i:i+size, j:j+size]
            median_value = np.median(window)
            filtered_img[i, j] = median_value

    return filtered_img.astype(np.uint8)
