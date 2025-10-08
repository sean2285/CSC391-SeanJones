import cv2
import numpy as np
from calculate_gradient import calculate_gradient

def sobel_edge_detector(img, threshold=100):

    grad_magnitude, _ = calculate_gradient(img)
    edge_map = np.zeros_like(grad_magnitude)
    edge_map[grad_magnitude > threshold] = 255
    return edge_map