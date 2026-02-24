import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(42)

x = np.linspace(0, 10, 100)

n_distractors = np.random.randint(1, 4)
colors = plt.cm.tab10.colors

plt.figure(figsize=(10, 6))

# Generate distractor curves as composite sin/cos functions
for i in range(n_distractors):
    a, b = np.random.uniform(0.5, 2, 2)
    phase = np.random.uniform(0, np.pi)
    if (i + 1) % 2 == 0:
        y_distractor = a * np.sin(x + phase) + b * np.cos(x * 0.7 + phase * 0.5)**2 + np.sin(x)**2
    else:
        y_distractor = a * np.cos(x + phase) + b * np.sin(x * 0.5 + phase * 0.3)**2 + np.cos(x)**2

    plt.plot(x, y_distractor, label=chr(65 + i), color=colors[i])

# Create two correlated curves with slight noise
base_curve = 2 * np.cos(x) + 1
noise_a = np.random.normal(0, 0.15, len(x))
noise_b = np.random.normal(0, 0.15, len(x))
curve_a = base_curve - 3 + noise_a
curve_b = base_curve + 2 + noise_b

label_a = chr(65 + n_distractors)
label_b = chr(65 + n_distractors + 1)

plt.plot(x, curve_a, label=label_a, color=colors[n_distractors], linewidth=2.5)
plt.plot(x, curve_b, label=label_b, color=colors[n_distractors + 1], linewidth=2.5, linestyle='--')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()