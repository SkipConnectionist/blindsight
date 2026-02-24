import numpy as np
import matplotlib.pyplot as plt

# Choose 4 well-spaced height values (y-axis)
heights = [2, 5, 8, 11]

# For each height, randomly sample count between 2-4 and create bars
np.random.seed(42)
bars = []
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
label_idx = 0

for h_val in heights:
    count = np.random.randint(2, 5)
    for i in range(count):
        # Assign x position for each bar
        x_pos = label_idx
        bars.append((x_pos, h_val, labels[label_idx]))
        label_idx += 1

# Shuffle the bars
np.random.shuffle(bars)

# Extract data for plotting
x_positions = [idx for idx, _ in enumerate(bars)]
y_heights = [b[1] for b in bars]
bar_labels = [b[2] for b in bars]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
bars_plot = ax.bar(x_positions, y_heights, width=0.6, color='steelblue')

# Remove y-axis and title
ax.set_yticks([])
ax.set_ylabel('')
ax.set_title('')
ax.set_xticks(x_positions)
ax.set_xticklabels(bar_labels)

plt.tight_layout()
plt.show()