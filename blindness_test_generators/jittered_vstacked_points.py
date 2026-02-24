import matplotlib.pyplot as plt
import numpy as np

n_circles = 8
alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:n_circles]

fig, ax = plt.subplots(figsize=(8, 8))

for i, letter in enumerate(alphabet):
    y = n_circles - i - 1
    x = np.random.uniform(0.3, 0.7) + np.random.normal(0.1, 0.02)

    circle = plt.Circle((x, y), 0.3, color='skyblue', alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, y, letter, ha='center', va='center', fontsize=14, fontweight='bold')

ax.set_xlim(0, n_circles - 1)
ax.set_ylim(-0.5, n_circles - 0.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()