from nupic.torch.sdr import SparseDistributedRepresentation

# Create a 2048-bit SDR with 20 active bits
sdr = SparseDistributedRepresentation(size=2048, active_count=20)
print("SDR shape:", sdr.size)
print("Active bits:", sdr.active_indices)