import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------
# Parameters
# -----------------------
sdr_dim = 128         # Size of the SDR vector
sparsity = 10         # Number of active bits per SDR
num_patterns = 5      # Number of different input sequences
seq_len = 3           # Length of each sequence
num_chunks = 3        # Number of clusters (chunks) to detect

# -----------------------
# Helper: Generate SDR
# -----------------------
def generate_sdr():
    sdr = np.zeros(sdr_dim, dtype=np.float32)
    sdr[np.random.choice(sdr_dim, sparsity, replace=False)] = 1.0
    return sdr

# -----------------------
# Build synthetic dataset
# -----------------------
patterns = [[generate_sdr() for _ in range(seq_len)] for _ in range(num_patterns)]
X_data = []
Y_data = []

for pat in patterns:
    for i in range(seq_len - 1):
        X_data.append(pat[i])
        Y_data.append(pat[i + 1])

X = torch.tensor(X_data)
Y = torch.tensor(Y_data)

# -----------------------
# Hebbian Layer
# -----------------------
class HebbianLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(dim, dim), requires_grad=False)

    def forward(self, x):
        return torch.sigmoid(x @ self.weights)

    def hebbian_update(self, pre, post, lr=0.01):
        delta_w = lr * (pre.T @ post)
        self.weights.data += delta_w

# -----------------------
# Predictive Model using Hebbian Layer
# -----------------------
class HebbianPredictiveModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = HebbianLayer(dim)

    def forward(self, x):
        return self.layer(x)

    def update_hebbian(self, x, target, lr=0.01):
        self.layer.hebbian_update(x, target, lr=lr)

# -----------------------
# Loss Function
# -----------------------
def sdr_overlap_loss(pred, target):
    overlap = (pred * target).sum(dim=1)
    norm = target.sum(dim=1) + 1e-6
    return (1.0 - overlap / norm).mean()

# -----------------------
# Train the model
# -----------------------
model = HebbianPredictiveModel(sdr_dim)
losses = []

for epoch in range(100):
    pred = model(X)
    loss = sdr_overlap_loss(pred, Y)
    losses.append(loss.item())
    model.update_hebbian(X, Y, lr=0.05)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# -----------------------
# Chunk Detection (KMeans)
# -----------------------
kmeans = KMeans(n_clusters=num_chunks, random_state=42)
chunk_labels = kmeans.fit_predict(X_data)
chunk_centers = kmeans.cluster_centers_

# -----------------------
# Visualize with PCA
# -----------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_data)
centers_pca = pca.transform(chunk_centers)

plt.figure(figsize=(8,6))
for i in range(num_chunks):
    points = X_pca[chunk_labels == i]
    plt.scatter(points[:, 0], points[:, 1], label=f'Chunk {i}', alpha=0.6)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=100, label='Centers')
plt.title("Chunk Detection via KMeans on SDRs (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
