import cv2
import numpy as np

def calculate_gradient(img):
    # Sobel filters
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Apply convolution directly using OpenCV
    gx = cv2.filter2D(img, cv2.CV_32F, Sx)
    gy = cv2.filter2D(img, cv2.CV_32F, Sy)

    # Gradient magnitude
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)
    
    return grad_mag
