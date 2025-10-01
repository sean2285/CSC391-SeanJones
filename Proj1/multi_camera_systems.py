import numpy as np
import rawpy
import cv2
import math
import matplotlib.pyplot as plt
import os

def horizontal_fov(focal_length_mm, sensor_width_mm):
    return 2 * math.degrees(math.atan(sensor_width_mm / (2 * focal_length_mm)))

def load_dng_as_rgb(path):
    # Check if the file exists before trying rawpy
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8
            )
        return rgb
    except Exception as e:
        raise RuntimeError(f"⚠️ Could not open {path} with rawpy. Error: {e}")

def compute_region_stats(img, x0, y0, w, h):
    region = img[y0:y0+h, x0:x0+w]
    if region.ndim == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    else:
        region_gray = region
    mu = np.mean(region_gray)
    sigma = np.std(region_gray)
    return mu, sigma

if __name__ == "__main__":
    f_main = 24.0
    w_main = 9.8
    f_tele = 77.0
    w_tele = 4.0

    fov_main = horizontal_fov(f_main, w_main)
    fov_tele = horizontal_fov(f_tele, w_tele)
    print(f"Horizontal FOV (main): {fov_main:.2f}°")
    print(f"Horizontal FOV (tele): {fov_tele:.2f}°")

    
    main_path = "/Users/seanjones/Desktop/maincamera.dng"
    tele_path = "/Users/seanjones/Desktop/telephotocamera.dng"

    main_img = load_dng_as_rgb(main_path)
    tele_img = load_dng_as_rgb(tele_path)

    roi = (100, 100, 50, 50)

    mu_main, sigma_main = compute_region_stats(main_img, *roi)
    mu_tele, sigma_tele = compute_region_stats(tele_img, *roi)

    print(f"Main camera (ROI): μ = {mu_main:.2f}, σ = {sigma_main:.2f}")
    print(f"Telephoto camera (ROI): μ = {mu_tele:.2f}, σ = {sigma_tele:.2f}")

    main_disp = main_img.copy()
    tele_disp = tele_img.copy()
    cv2.rectangle(main_disp, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0,255,0), 2)
    cv2.rectangle(tele_disp, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0,255,0), 2)

    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].imshow(main_disp)
    axes[0].set_title("Main camera ROI")
    axes[1].imshow(tele_disp)
    axes[1].set_title("Telephoto ROI")
    plt.show()
