import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def y1_func(x):
    return 2 * np.sin(x) + 6 - 0.4 * x


def y2_func(x):
    return 0.08 * x**2 + 0.3 * x + 1


def intersection_eq(x):
    return y1_func(x) - y2_func(x)


x_intersection = fsolve(intersection_eq, 12)[0]
y_intersection = y1_func(x_intersection)

print(f"Intersection at x = {x_intersection:.4f}, y = {y_intersection:.4f}")

# Set limits with appropriate margins
x_limit = x_intersection * 0.93
x = np.linspace(0, x_limit, 200)
y1 = y1_func(x)
y2 = y2_func(x)

# Calculate y limits to show both curves properly
y_min = min(np.min(y1), np.min(y2))
y_max = max(np.max(y1), np.max(y2))
y_margin = (y_max - y_min) * 0.1

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='NotGreatCurve', linewidth=2, color='blue')
plt.plot(x, y2, label='SpectacularCurve', linewidth=2, color='red')
plt.xlim(0, x_limit)
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()