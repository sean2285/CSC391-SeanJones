import cv2
import numpy as np
from calculate_gradient import calculate_gradient

def directional_edge_detector(img, direction_range=(40, 50)):
   
    _, grad_angle = calculate_gradient(img)

    min_dir, max_dir = direction_range
    edge_directional_map = np.zeros_like(grad_angle, dtype=np.uint8)
    mask = (grad_angle >= min_dir) & (grad_angle <= max_dir)
    edge_directional_map[mask] = 255
    return edge_directional_map