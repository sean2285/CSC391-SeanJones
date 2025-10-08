import cv2
import numpy as np

def calculate_gradient(img):
    
    # Sobel kernels
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Compute gradients
    gx = cv2.filter2D(img, cv2.CV_32F, Sx)
    gy = cv2.filter2D(img, cv2.CV_32F, Sy)

    # Gradient magnitude and direction
    grad_magnitude = np.sqrt(gx**2 + gy**2)
    grad_angle = np.arctan2(gy, gx) * (180 / np.pi)  # Convert radians → degrees

    # Normalize and convert to 8-bit image
    grad_magnitude = np.clip(grad_magnitude, 0, 255).astype(np.uint8)

    return grad_magnitude, grad_angle

