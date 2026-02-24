import matplotlib.pyplot as plt
import numpy as np
import random

fig, ax = plt.subplots()
ax.set_axis_off()

np.random.seed(42)

n_lines = 6
lines = []

parallel_idx = np.random.choice(n_lines, 2, replace=False)
parallel_slope = np.random.uniform(-2, 2)

for i in range(n_lines):
    if i in parallel_idx:
        slope = parallel_slope
    else:
        slope = np.random.uniform(-2, 2)

    x1 = np.random.uniform(0, 3)
    x2 = np.random.uniform(3, 6)
    y1 = np.random.uniform(0, 4)
    y2 = y1 + slope * (x2 - x1)

    lines.append(((x1, y1), (x2, y2)))

random.shuffle(lines)

for i, (start, end) in enumerate(lines):
    x1, y1 = start
    x2, y2 = end
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    ax.text(x2 + 0.1, y2, chr(ord('A') + i), fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()