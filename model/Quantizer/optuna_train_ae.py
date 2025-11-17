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
import optuna

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base configuration
CONFIG = {
    "random_seed": 42,
    "batch_size": 1024,  # Reduced for stability
    "num_training_updates": 15000,
    "model_save_path": "vqvaemodel.pth",  # Save only one model
    "model_params": {
        "num_hiddens": 64,
        "num_residual_hiddens": 64,
        "num_residual_layers": 3,
        "embedding_dim": 128,
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

# Prepare train, validation, and test data loaders
def prepare_data_loaders(train_idxs, test_idxs, X, Y, groups, batch_size):
    tmp_X_train, X_test = X[train_idxs], X[test_idxs]
    tmp_Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    group_train = groups[train_idxs]

    final_train_idxs, final_val_idxs = next(LeaveOneGroupOut().split(tmp_X_train, tmp_Y_train, groups=group_train))
    X_train, X_val = tmp_X_train[final_train_idxs], tmp_X_train[final_val_idxs]
    Y_train, Y_val = tmp_Y_train[final_train_idxs], tmp_Y_train[final_val_idxs]

    train_dataset = NormalDataset(X_train, Y_train, name="train", isLabel=True)
    val_dataset = NormalDataset(X_val, Y_val, name="val", isLabel=True)
    test_dataset = NormalDataset(X_test, Y_test, pid=groups[test_idxs], name="test", isLabel=True)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True),
    )

# Train the VQ-VAE model and return validation reconstruction error
def train_model(X, Y, P, config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data_variance = torch.var(torch.tensor(X, dtype=torch.float32, device=device))

    folds = LeaveOneGroupOut().split(X, Y, groups=P)
    train_res_recon_error, val_res_recon_error = [], []

    # Select the first fold only
    train_idxs, test_idxs = next(folds)
    logging.info(f"Training on one fold only.")

    train_loader, val_loader, _ = prepare_data_loaders(train_idxs, test_idxs, X, Y, P, config["batch_size"])

    # Initialize model and optimizer
    model = VQ_VAE_Model(**config["model_params"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for update_idx in range(config["num_training_updates"]):
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, _ = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss

            loss.backward()
            optimizer.step()

            train_res_recon_error.append(recon_error.item())

        # Log progress every 100 updates
        if (update_idx + 1) % 100 == 0:
            avg_recon = np.mean(train_res_recon_error[-100:])
            logging.info(f"Update {update_idx + 1}: Recon Error: {avg_recon:.3f}")

    # Validation phase
    model.eval()
    val_errors = []
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            vq_loss, data_recon, _ = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            val_errors.append(recon_error.item())

    avg_val_recon = np.mean(val_errors)
    val_res_recon_error.append(avg_val_recon)

    logging.info(f"Validation Recon Error: {avg_val_recon:.3f}")

    # Save the model
    model_path = config["model_save_path"]
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved at {model_path}")

    return avg_val_recon  # Return validation error

# Optuna objective function
def objective(trial):
    # Suggest a window size in a reasonable range
    window_size = trial.suggest_int("input_length", 50, 300, step=10)

    # Update configuration
    config = CONFIG.copy()
    config["input_length"] = window_size

    # Load dataset
    root_path = "/home/lala/Downloads/"
    dataset = "oppo_33hz_w10_o5"
    X = np.load(f"{root_path}/{dataset}/X.npy")
    Y = np.load(f"{root_path}/{dataset}/Y.npy")
    P = np.load(f"{root_path}/{dataset}/pid.npy")

    # Resize data if needed
    if X.shape[1] != config["input_length"]:
        X = resize_data(X, config["input_length"])
    X = X.astype("float32")
    X = np.transpose(X, (0, 2, 1))  # Convert to channels-first format

    # Train model and return validation error
    val_recon_error = train_model(X, Y, P, config)
    return val_recon_error  # Minimize validation reconstruction error

# Run Optuna to find the best window size
if __name__ == "__main__":
    set_seed(CONFIG["random_seed"])
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Print and log the best result
    best_window_size = study.best_params['input_length']
    logging.info(f"Best window size: {best_window_size}")
    print(f"Best window size: {best_window_size}")
