import numpy as np
import matplotlib.pyplot as plt

# Global parameters
signal_freq = 5.0      # Hz
duration = 2           # seconds
sampling_freq = 8      # Hz
num_bits = 3           # number of bits (2^3 = 8 levels)
min_signal = -1        # minimum signal value
max_signal = 1         # maximum signal value
mean = 0               # Gaussian noise mean
std_dev = 0.1          # Gaussian noise std deviation

# Original signal
def original_signal(t):
    return np.sin(2 * np.pi * signal_freq * t)

# Add Gaussian noise
def add_Gaussian_noise(signal, mean, std):
    mag = np.max(signal) - np.min(signal)
    noise = np.random.normal(mean, std * mag, len(signal))
    return signal + noise

# Quantization
def quantize(signal, num_bits, min_signal, max_signal):
    L = 2**num_bits
    q_s = np.round((signal - min_signal) / (max_signal - min_signal) * (L - 1))
    q_v = min_signal + q_s * (max_signal - min_signal) / (L - 1)
    return q_v

# Error metrics
def compute_errors(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10((np.max(np.abs(original)) ** 2) / mse) if mse != 0 else np.inf
    return mse, rmse, psnr

# Generate continuous signal
t_points = np.linspace(0, duration, 1000, endpoint=False)
cont_signal = original_signal(t_points)

# Sample original signal
n = int(sampling_freq * duration)
t_sampled = np.linspace(0, duration, n, endpoint=False)
sampled_signal = original_signal(t_sampled)

# Add noise
noisy_signal = add_Gaussian_noise(sampled_signal, mean, std_dev)

# Quantize noisy signal
quantized_noisy = quantize(noisy_signal, num_bits, min_signal, max_signal)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t_points, cont_signal, label="Continuous Signal", color="blue")
plt.scatter(t_sampled, noisy_signal, color="orange", label="Sampled + Noise")
plt.step(t_sampled, quantized_noisy, where="post",
         label=f"Quantized Noisy Signal ({num_bits} bits)",
         color="red", linestyle="--")

plt.title("Noise and Error Analysis in Sampling & Quantization")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Compute and print errors
mse, rmse, psnr = compute_errors(sampled_signal, noisy_signal)
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"PSNR = {psnr:.2f} dB")
