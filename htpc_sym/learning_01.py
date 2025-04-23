import numpy as np
import matplotlib.pyplot as plt

# Symbolic HTPC on real sequences from a local text file
# Each line in 'sequences.txt' contains a whitespace-separated token sequence

data_file = 'sequences.txt'  # replace with your file path

# Load and preprocess sequences
token_seqs = []
with open(data_file) as f:
    for line in f:
        seq = line.strip().split()
        if seq:
            token_seqs.append(seq)

# Build vocabulary
vocab = sorted({tok for seq in token_seqs for tok in seq})
A = len(vocab)
token2idx = {tok: i for i, tok in enumerate(vocab)}

# HTPC parameters
C = 6           # number of chunk units
epochs = 100
alpha = 0.1     # inference step size (bottom-up weight)
beta = 0.9      # temporal persistence of chunk belief
eta = 0.01      # learning rate

# Initialize top-down weights U (A x C)
np.random.seed(0)
U = np.random.randn(A, C) * 0.1

# Training across sequences, resetting r1 at each sequence boundary
mse_history = []
for ep in range(epochs):
    total_mse = 0.0
    total_tokens = 0
    for seq in token_seqs:
        r1 = np.zeros(C)  # reset chunk belief at start of each sequence
        for tok in seq:
            # one-hot input vector r0
            r0 = np.zeros(A)
            r0[token2idx[tok]] = 1
            # inference: update r1 from previous state + current input
            for _ in range(5):
                err = r0 - U.dot(r1)
                r1 = beta*r1 + alpha * U.T.dot(err)
            # record error
            pred = U.dot(r1)
            total_mse += np.mean((r0 - pred)**2)
            total_tokens += 1
            # learning
            U += eta * np.outer((r0 - pred), r1)
    mse_history.append(total_mse / total_tokens)

# Plot reconstruction error\ nplt.figure(figsize=(8, 4))
plt.plot(mse_history, color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title(f'Temporal Symbolic HTPC: Error (resets at seq boundaries)')
plt.grid(True)
plt.show()

# Display vocabulary
print('Vocabulary:', vocab)

# Show learned chunk prototypes
print('\nChunk weight matrix U:')
for i, tok in enumerate(vocab):
    print(f"{tok:>10}: {U[i].round(3)}")

# Top-3 tokens per chunk
for j in range(C):
    top = np.argsort(U[:, j])[-3:]
    print(f"Chunk {j} top tokens: {[vocab[i] for i in top]}")

# Visualize chunks\ nfig, axes = plt.subplots(1, C, figsize=(4*C, 4))
for j, ax in enumerate(axes):
    ax.bar(vocab, U[:, j])
    ax.set_title(f"Chunk {j}")
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
