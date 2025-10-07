import cv2
import numpy as np

def contrast_stretch(img, r_min, r_max): 

# Apply linear contrast stretching
    stretched = (img - r_min) * (255.0 / (r_max - r_min))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched
