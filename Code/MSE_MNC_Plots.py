import json
import matplotlib.pyplot as plt
import numpy as np

# Load the statistics
with open('visualization/global_statistics.json', 'r') as f:
    stats = json.load(f)

# Extract MSE and MNC values
mse_values = [(entry['batch'], entry['mse']) for entry in stats['mse_values']]
mnc_values = [(entry['batch'], entry['mnc']) for entry in stats['mnc_values']]

# Sort by batch number
mse_values.sort(key=lambda x: x[0])
mnc_values.sort(key=lambda x: x[0])

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# MSE plot
batches, mse = zip(*mse_values)
ax1.plot(batches, mse, 'b-', label='MSE')
ax1.set_xlabel('Batch')
ax1.set_ylabel('MSE')
ax1.set_title('MSE over Batches')
ax1.grid(True)

# MNC plot
batches, mnc = zip(*mnc_values)
ax2.plot(batches, mnc, 'r-', label='MNC')
ax2.set_xlabel('Batch')
ax2.set_ylabel('MNC')
ax2.set_title('MNC over Batches')
ax2.grid(True)

plt.tight_layout()
plt.show()
plt.close()
