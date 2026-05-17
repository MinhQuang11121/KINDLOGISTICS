import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Load the traffic flow matrix
X = np.load('traffic_flow.npy')
print(f"Loaded data shape: {X.shape}")

# Parameters
num_timesteps_input = 12  # 3 hours (12 * 15 min)
num_timesteps_output = 3   # 45 minutes (3 * 15 min)
total_steps = X.shape[0]
max_start_idx = total_steps - num_timesteps_input - num_timesteps_output

# Calculate mean and std from training set (first 70%)
train_end_idx = int(0.7 * max_start_idx)
train_data = X[:train_end_idx + num_timesteps_input + num_timesteps_output]  # Include enough for normalization
mean = train_data.mean()
std = train_data.std()
print(f"Training set mean: {mean:.4f}, std: {std:.4f}")

# Normalize the entire dataset
X_norm = (X - mean) / std

# Define the Dataset class
class TrafficDataset(Dataset):
    def __init__(self, data, indices, input_len=12, output_len=3):
        self.data = data
        self.indices = indices
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        input_seq = self.data[start:start + self.input_len]
        output_seq = self.data[start + self.input_len:start + self.input_len + self.output_len]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# Split indices: 70% train, 10% val, 20% test
train_indices = list(range(0, int(0.7 * max_start_idx)))
val_indices = list(range(int(0.7 * max_start_idx), int(0.8 * max_start_idx)))
test_indices = list(range(int(0.8 * max_start_idx), max_start_idx))

print(f"Train samples: {len(train_indices)}")
print(f"Val samples: {len(val_indices)}")
print(f"Test samples: {len(test_indices)}")

# Create datasets
train_dataset = TrafficDataset(X_norm, train_indices, num_timesteps_input, num_timesteps_output)
val_dataset = TrafficDataset(X_norm, val_indices, num_timesteps_input, num_timesteps_output)
test_dataset = TrafficDataset(X_norm, test_indices, num_timesteps_input, num_timesteps_output)

# Create DataLoaders with batching
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Get a batch
if __name__ == "__main__":
    for inputs, targets in train_loader:
        print(f"Input batch shape: {inputs.shape}")  # (batch_size, 12, 6567)
        print(f"Target batch shape: {targets.shape}")  # (batch_size, 3, 6567)
        break
