import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

data = [
    np.concatenate([np.random.normal(0, 1, 50), np.array([4, -5, 3.5, -4, 5])]),
    np.concatenate([np.random.normal(2, 1.5, 50), np.array([8, -6, 9])]),
    np.concatenate([np.random.normal(-1, 0.8, 50), np.array([-4.5, 3.5, 4, -5])])
]

plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['Group A', 'Group B', 'Group C'])
plt.ylabel('Values')
plt.show()