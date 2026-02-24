import matplotlib.pyplot as plt
import random

num_points = random.randint(5, 10)
colors = plt.cm.tab20(random.sample(range(20), num_points))
letters = [chr(65 + i) for i in range(num_points)]

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

for i in range(num_points):
    x = random.uniform(0.5, 9.5)
    y = random.uniform(0.5, 9.5)
    ax.scatter(x, y, color=colors[i], s=200, label=f'Point {letters[i]}', zorder=5)
    ax.text(x, y, letters[i], ha='center', va='center', color='white', fontweight='bold', zorder=6)

ax.legend(loc='best')
ax.axis('off')
plt.tight_layout()
plt.show()