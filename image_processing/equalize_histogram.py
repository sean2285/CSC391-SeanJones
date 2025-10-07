import cv2
import numpy as np

def equalize_histogram(img):
   
    # Compute histogram
    counts, bins = np.histogram(img.flatten(), bins=256, range=[0, 255])

    # Compute cumulative distribution function (CDF)
    cdf = counts.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # Map pixel values using the equalized CDF
    new_img = cdf_normalized[img]
    return new_img
