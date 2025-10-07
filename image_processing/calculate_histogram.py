import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(img, bins=256):
    
    counts, bin_edges = np.histogram(img.flatten(), bins=bins, range=[0, 255])
    dist = counts / counts.sum()
    return counts, dist
