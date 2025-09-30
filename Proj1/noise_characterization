import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os

# Dark Frame Files
dark_frame_files = [
    "darkframe1.dng",
    "darkframe2.dng",
    "darkframe3.dn"
]

# Output folder for plots
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Analysis Function
def analyze_dark_frame(file_path, patch_size=100):
    with rawpy.imread(file_path) as raw:
        img = raw.raw_image_visible.astype(np.float32)

    # Basic info
    print(f"\nFile: {file_path}")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Min: {img.min()}, Max: {img.max()}")

    # Pick a central 100x100 patch
    h, w = img.shape
    start_x = w // 2 - patch_size // 2
    start_y = h // 2 - patch_size // 2
    patch = img[start_y:start_y+patch_size, start_x:start_x+patch_size]

    mean_val = np.mean(patch)
    std_val = np.std(patch)

    print(f"  Patch mean (μ): {mean_val:.2f}")
    print(f"  Patch std (σ): {std_val:.2f}")

    # Histogram for the patch
    plt.figure(figsize=(6,4))
    plt.hist(patch.flatten(), bins=50, color="steelblue", edgecolor="black")
    plt.title(f"Histogram - {os.path.basename(file_path)}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"hist_{os.path.basename(file_path)}.png")
    plt.savefig(out_file)
    plt.close()

    return mean_val, std_val

# Run analysis on each dark frame
results = []
for f in dark_frame_files:
    if os.path.exists(f):
        μ, σ = analyze_dark_frame(f)
        results.append((f, μ, σ))
    else:
        print(f"File not found: {f}")

# Results
print("\n=== Dark Frame Results ===")
print(f"{'File':20s} {'Mean (μ)':>10s} {'Std (σ)':>10s}")
for fname, μ, σ in results:
    print(f"{os.path.basename(fname):20s} {μ:10.2f} {σ:10.2f}")

# Histogram Comparison
plt.figure(figsize=(7,5))
for f in dark_frame_files:
    if os.path.exists(f):
        with rawpy.imread(f) as raw:
            img = raw.raw_image_visible.astype(np.float32)
        h, w = img.shape
        patch = img[h//2-50:h//2+50, w//2-50:w//2+50]
        plt.hist(patch.flatten(), bins=60, alpha=0.5, label=os.path.basename(f))

plt.title("Dark Frame Patch Histograms Comparison")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_histograms.png"))
plt.close()

