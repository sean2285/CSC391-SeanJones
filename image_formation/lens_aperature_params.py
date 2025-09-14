import numpy as np
import matplotlib.pyplot as plt

def thin_lens_zi(f, z0):
    """Compute image distance zi using thin lens law."""
    return 1 / (1/f - 1/z0)

# Object distances: 1.1f to 10^4 mm, 4 points per mm
z0_min = 1.1
z0_max = 1e4
points_per_mm = 4
z0 = np.linspace(z0_min, z0_max, int((z0_max-z0_min)*points_per_mm))

focal_lengths = [3, 9, 50, 200]  # mm

plt.figure(figsize=(8,6))
for f in focal_lengths:
    zi = thin_lens_zi(f, z0)
    plt.loglog(z0, zi, label=f"f = {f} mm")
    plt.axvline(f, color='k', linestyle='--', linewidth=1)

plt.xlabel("Object distance $z_0$ (mm)")
plt.ylabel("Image distance $z_i$ (mm)")
plt.ylim([0, 3000])
plt.legend()
plt.grid(True, which='both', linestyle=':', linewidth=0.7)
plt.title("Thin Lens Law: $z_i$ vs $z_0$")

plt.show()

# Aperture formula D = f / N
real_world_lenses = [
    (24, 1.4),
    (50, 1.8),
    (70, 2.8),
    (200, 2.8),
    (400, 2.8),
    (600, 4.0)
]

print("Aperture diameters (D = f/N):")
for f, N in real_world_lenses:
    D = f / N
    print(f"f = {f} mm, N = {N} → D = {D:.2f} mm")