import cv2
import numpy as np
from apply_convolution import apply_convolution

def calculate_gradient(img):
   
    # Define Sobel kernels
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Apply Sobel filters
    Gx = apply_convolution(img, Sx)
    Gy = apply_convolution(img, Sy)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)
    grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)

    return grad_magnitude
