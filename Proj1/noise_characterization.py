import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os

# Dark Frame Files
dark_frame_files = [
    'images/darkframe1.dng',
    'images/darkframe2.dng',
    'images/darkframe3.dng'
]

# Output folder for plots
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Run analysis and collect patches
patches = []
for f in dark_frame_files:
    if os.path.exists(f):
        with rawpy.imread(f) as raw:
            img = raw.raw_image_visible.astype(np.float32)
        h, w = img.shape
        patch = img[h//2-50:h//2+50, w//2-50:w//2+50]
        patches.append((os.path.basename(f), patch.flatten()))
    else:
        print(f"File not found: {f}")

# Histogram of all patches combined
plt.figure(figsize=(7,5))
for name, patch_data in patches:
    plt.hist(patch_data, bins=60, alpha=0.5, label=name)

plt.title("Histogram of Dark Frames")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(output_dir, "combined_darkframe_histogram.png"))
plt.show()

