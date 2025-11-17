import numpy as np
import torch
import matplotlib.pyplot as plt
from imu_test import Model as VQ_VAE_Model
from train_ae import set_seed, resize_data, CONFIG  # Reusing from train script

# Set random seed
set_seed(CONFIG["random_seed"])

# Load dataset
root_path = "/home/lala/Downloads/"
dataset = "oppo_33hz_w10_o5"
X = np.load(f"{root_path}/{dataset}/X.npy")

# Resize and preprocess data
if X.shape[1] != CONFIG["input_length"]:
    X = resize_data(X, CONFIG["input_length"])
X = X.astype("float32")
X = np.transpose(X, (0, 2, 1))  # Convert to channels-first format

# Select a single data point for visualization
sample_idx = 0  # Change index to test different samples
input_data = torch.tensor(X[sample_idx:sample_idx+1], dtype=torch.float32)

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VQ_VAE_Model(**CONFIG["model_params"]).to(device)
model_path = CONFIG["model_save_path"].format(fold=1)  # Change fold if needed
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Run inference
with torch.no_grad():
    vq_loss, reconstructed_data, _ = model(input_data.to(device))
    reconstructed_data = reconstructed_data.cpu().numpy()

# Plot original vs. reconstructed signal
fig, ax = plt.subplots(figsize=(10, 5))
time_axis = np.arange(CONFIG["input_length"])
ax.plot(time_axis, X[sample_idx][0], label="Original", linestyle="--", alpha=0.7)
ax.plot(time_axis, reconstructed_data[0, 0], label="Reconstructed", alpha=0.9)
ax.set_title("Original vs. Reconstructed Signal")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Signal Amplitude")
ax.legend()
plt.show()
