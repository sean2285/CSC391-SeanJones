import numpy as np
import matplotlib.pyplot as plt

signal_freq = 5.0     # Hz
duration = 2           # seconds
sampling_freq = 8      # Hz
num_bits = 3           # number of bits (2^3 = 8 levels)
min_signal = -1        # minimum signal value
max_signal = 1         # maximum signal value

def original_signal(t):
    return np.sin(2 * np.pi * signal_freq * t)

t_points = np.linspace(0, duration, 1000, endpoint=False)
cont_signal = original_signal(t_points)

plt.figure(figsize=(12, 6))
plt.plot(t_points, cont_signal, label="Continuous Signal", color="blue")

n = int(sampling_freq * duration)
t_sampled = np.linspace(0, duration, n, endpoint=False)
sampled_signal = original_signal(t_sampled)

plt.scatter(t_sampled, sampled_signal, color="black", label="Sampled Points")

L = 2**num_bits
q_s = np.round((sampled_signal - min_signal) / (max_signal - min_signal) * (L - 1))
q_v = min_signal + q_s * (max_signal - min_signal) / (L - 1)

plt.step(t_sampled, q_v, where="post",
         label=f"Quantized Signal ({num_bits} bits)",
         color="red", linestyle="--")

plt.title("Sampling and Quantization of a 1D Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

mse = np.mean((sampled_signal - q_v) ** 2)
print(f"Mean Squared Error (Quantization): {mse:.4f}")