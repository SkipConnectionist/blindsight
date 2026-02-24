import matplotlib.pyplot as plt
import numpy as np


def plot_random_shapes():
    fig, ax = plt.subplots(figsize=(10, 8))

    shapes = ['star', 'diamond', 'circle', 'square']
    counts = {shape: np.random.randint(2, 8) for shape in shapes}

    for i, (shape, count) in enumerate(counts.items()):
        x = np.random.uniform(0, 10, count)
        y = np.random.uniform(0, 10, count)
        sizes = np.random.uniform(50, 200, count)

        if shape == 'star':
            marker = '*'
        elif shape == 'diamond':
            marker = 'D'
        elif shape == 'circle':
            marker = 'o'
        elif shape == 'square':
            marker = 's'

        ax.scatter(x, y, s=sizes, marker=marker,
                   label=f'{shape.capitalize()}', alpha=0.7)

        print(f'Count for {shape} = {count}')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend()
    plt.tight_layout()
    plt.show()


plot_random_shapes()