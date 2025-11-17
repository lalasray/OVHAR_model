import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import random
import logging
from imu_test import Model as VQ_VAE_Model
from sslearning.data.data_loader import NormalDataset
import torch.optim as optim
import collections

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
CONFIG = {
    "random_seed": 42,
    "input_length": 64,
    "batch_size": 1024,  # Reduced for stability
    "num_training_updates": 15000,
    "model_save_path": "vqvaemodel_wisdom_512.pth",
    "model_params": {
        "num_hiddens": 64,
        "num_residual_hiddens": 64,
        "num_residual_layers": 3,
        "embedding_dim": 512,  #36,96,256,512
        "num_embeddings": 1028,
        "commitment_cost": 0.25,
        "decay": 0.99,
    },
    "learning_rate": 1e-3,
}

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Faster for consistent input sizes

# Resize data using linear interpolation
def resize_data(X, target_length, axis=1):
    t_orig = np.linspace(0, 1, X.shape[axis], endpoint=True)
    t_new = np.linspace(0, 1, target_length, endpoint=True)
    return interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(t_new)

# Calculate class weights for imbalanced datasets
def calculate_class_weights(labels):
    counter = collections.Counter(labels)
    total_samples = len(labels)
    weights = [1.0 / (counter.get(i, 1) / total_samples) for i in range(max(counter.keys()) + 1)]
    logging.info(f"Class weights: {weights}")
    return weights

# Prepare train, validation, and test data loaders
def prepare_data_loaders(train_idxs, test_idxs, X, Y, groups, batch_size):
    tmp_X_train, X_test = X[train_idxs], X[test_idxs]
    tmp_Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    group_train = groups[train_idxs]

    final_train_idxs, final_val_idxs = next(LeaveOneGroupOut().split(tmp_X_train, tmp_Y_train, groups=group_train))
    X_train, X_val = tmp_X_train[final_train_idxs], tmp_X_train[final_val_idxs]
    Y_train, Y_val = tmp_Y_train[final_train_idxs], tmp_Y_train[final_val_idxs]

    # Save training dataset for later testing
    # np.save("X_train.npy", X_train)
    # np.save("Y_train.npy", Y_train)
    # np.save("P_train.npy", group_train[final_train_idxs])  # Save corresponding PIDs
    # logging.info("Training dataset saved.")

    train_dataset = NormalDataset(X_train, Y_train, name="train", isLabel=True)
    val_dataset = NormalDataset(X_val, Y_val, name="val", isLabel=True)
    test_dataset = NormalDataset(X_test, Y_test, pid=groups[test_idxs], name="test", isLabel=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True),
    )

# Train the VQ-VAE model
def train_model(X, Y, P, config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data_variance = torch.var(torch.tensor(X, dtype=torch.float32, device=device))

    folds = LeaveOneGroupOut().split(X, Y, groups=P)
    train_res_recon_error, train_res_perplexity = [], []

    for fold_idx, (train_idxs, test_idxs) in enumerate(folds, start=1):
        logging.info(f"Starting fold {fold_idx}")
        train_loader, val_loader, test_loader = prepare_data_loaders(train_idxs, test_idxs, X, Y, P, config["batch_size"])

        # Initialize model and optimizer
        model = VQ_VAE_Model(**config["model_params"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Training loop
        for update_idx in range(config["num_training_updates"]):
            for data, _ in train_loader:  # Iterate properly over the dataset
                data = data.to(device)
                # print(data.shape)
                optimizer.zero_grad()

                vq_loss, data_recon, perplexity,_ = model(data)
                recon_error = F.mse_loss(data_recon, data) / data_variance
                loss = recon_error + vq_loss

                loss.backward()
                optimizer.step()

                train_res_recon_error.append(recon_error.item())
                train_res_perplexity.append(perplexity.item())

            # Log progress every 100 updates
            if (update_idx + 1) % 100 == 0:
                avg_recon = np.mean(train_res_recon_error[-100:])
                avg_perplexity = np.mean(train_res_perplexity[-100:])
                logging.info(f"Update {update_idx + 1}: Recon Error: {avg_recon:.3f}, Perplexity: {avg_perplexity:.3f}")

        # Save the model
        model_path = config["model_save_path"]
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        break

# Main execution
if __name__ == "__main__":
    set_seed(CONFIG["random_seed"])

    # Load dataset
    root_path = "/home/dfki/dee/VQKANClassifier-main/Datasets/"
    dataset = "wisdm_30hz_clean"
    X = np.load(f"{root_path}/{dataset}/X.npy")
    Y = np.load(f"{root_path}/{dataset}/Y.npy")
    P = np.load(f"{root_path}/{dataset}/pid.npy")

    print(X.shape)
    print(Y.shape)
    # Resize data if needed
    if X.shape[1] != CONFIG["input_length"]:
        X = resize_data(X, CONFIG["input_length"])
    X = X.astype("float32")
    X = np.transpose(X, (0, 2, 1))  # Convert to channels-first format

    # Train the model
    train_model(X, Y, P, CONFIG)
