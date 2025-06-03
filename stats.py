import torch
import statistics

# Load the list of floats from the .pt file
tensor = torch.load("clip_scores_100-200.pt")

# Compute statistics
min_val = min(tensor)
max_val = max(tensor)
mean_val = statistics.mean(tensor)
std_val = statistics.stdev(tensor)

# Print results
print(f"Min:  {min_val:.4f}")
print(f"Max:  {max_val:.4f}")
print(f"Mean: {mean_val:.4f}")
print(f"Std:  {std_val:.4f}")

sorted_tensor = sorted(tensor)
print("\nSorted values:")
for val in sorted_tensor:
    print(f"{val:.4f}")
