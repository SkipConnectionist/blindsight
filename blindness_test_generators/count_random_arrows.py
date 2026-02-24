import matplotlib.pyplot as plt
import numpy as np

directions = ['left', 'right', 'up', 'down']

fig, ax = plt.subplots()

for direction in directions:
    num_arrows = np.random.randint(1, 9)

    for _ in range(num_arrows):
        if direction == 'right':
            x, y = np.random.uniform(0.05, 0.85), np.random.uniform(0.05, 0.95)
            dx, dy = 0.15, 0
        elif direction == 'left':
            x, y = np.random.uniform(0.15, 0.95), np.random.uniform(0.05, 0.95)
            dx, dy = -0.15, 0
        elif direction == 'up':
            x, y = np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.85)
            dx, dy = 0, 0.15
        else:  # down
            x, y = np.random.uniform(0.05, 0.95), np.random.uniform(0.15, 0.95)
            dx, dy = 0, -0.15

        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.03)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.show()