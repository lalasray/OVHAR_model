import numpy as np
import matplotlib.pyplot as plt

# Parameters for synthetic data
num_samples = 1000   # Number of time steps
sampling_rate = 50   # Hz
time = np.arange(0, num_samples / sampling_rate, 1 / sampling_rate)

# Simulate accelerometer data (mimicking walking)
acc_x = 0.5 * np.sin(2 * np.pi * 1 * time) + np.random.normal(0, 0.02, num_samples)
acc_y = 0.5 * np.cos(2 * np.pi * 1 * time) + np.random.normal(0, 0.02, num_samples)
acc_z = 9.81 + np.random.normal(0, 0.02, num_samples)  # Constant gravity plus noise

# Combine accelerometer data
acc_data = np.stack([acc_x, acc_y, acc_z], axis=1)

# Plot the synthetic accelerometer data
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(time, acc_data[:, i])
    plt.title(['Acc X', 'Acc Y', 'Acc Z'][i])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')

plt.tight_layout()
plt.show()

