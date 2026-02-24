import numpy as np
import matplotlib.pyplot as plt

# Define the range
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# Sine curve (dotted)
y_sine = np.sin(x)

x0 = 0
m = np.cos(x0)
b = 5.5
y_line = m * x + b

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(x, y_sine, marker='o', linestyle='None', markersize=0.5,
         markevery=30, color='blue', label='alpha')
plt.plot(x, y_line, 'r-', label=f'Beta', linewidth=0.5)

# Distractor intersection point
# plt.plot(x0, np.sin(x0), 'ko', markersize=8)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()