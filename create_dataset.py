import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class CustomCSVDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path,header=None)  # Load CSV data
        print(self.data)
        self.transform = transform

        # Assume first column is the categorical string label
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data.iloc[:, 0])  # Convert to numerical labels

        # Extract coordinate columns (assume next 18 columns are x, y, z values)
        self.features = self.data.iloc[:, 1:].values.astype('float32')  # Convert to NumPy array

    def __len__(self):
        return len(self.data)  # Number of samples

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])  # Convert coordinates to tensor
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor

        if self.transform:
            x = self.transform(x)  # Apply transformations if any

        return x, y  # Return feature-label pair


# Example Usage
dataset = CustomCSVDataset("data.csv")

# DataLoader for batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check one batch
for batch in dataloader:
    inputs, targets = batch
    print(f"Inputs Shape: {inputs.shape}")  # Expected: (batch_size, 18)
    print(f"Targets Shape: {targets.shape}")  # Expected: (batch_size,)
    break

# Convert dataset into a dictionary and save
dataset_dict = {
    "features": torch.tensor(dataset.features),
    "labels": torch.tensor(dataset.labels)
}

torch.save(dataset_dict, "dataset.pt")  # Save to file
print("Dataset saved successfully!")

loaded_data = torch.load("dataset.pt")

# Extract features and labels
features = loaded_data["features"]
labels = loaded_data["labels"]

print("Loaded Features Shape:", features.shape)
print("Loaded Labels Shape:", labels.shape)

print(features[0,0]);