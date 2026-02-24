import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random


def generate_random_venn(sets=3, relationship_type='random'):
    """Generate random Venn diagram with specified relationship type.

    Args:
        sets: Number of sets (2-4)
        relationship_type: 'intersect', 'contained', 'no_overlap', 'slight_overlap', 'random'
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    positions = []
    radii = []

    legend_handles = []

    for i in range(sets):
        if relationship_type == 'no_overlap':
            angle = 2 * np.pi * i / sets
            x = 1.2 * np.cos(angle)
            y = 1.2 * np.sin(angle)
            radius = random.uniform(0.4, 0.5)

        elif relationship_type == 'contained':
            if i == 0:
                x, y = 0, 0
                radius = random.uniform(0.9, 1.1)
            else:
                angle = 2 * np.pi * (i - 1) / (sets - 1) if sets > 2 else 0
                x = 0.3 * np.cos(angle)
                y = 0.3 * np.sin(angle)
                radius = random.uniform(0.25, 0.35)

        elif relationship_type == 'slight_overlap':
            angle = 2 * np.pi * i / sets
            x = 0.7 * np.cos(angle)
            y = 0.7 * np.sin(angle)
            radius = random.uniform(0.45, 0.55)

        elif relationship_type == 'intersect':
            angle = 2 * np.pi * i / sets
            x = 0.4 * np.cos(angle)
            y = 0.4 * np.sin(angle)
            radius = random.uniform(0.5, 0.6)

        else:
            x = random.uniform(-0.8, 0.8)
            y = random.uniform(-0.8, 0.8)
            radius = random.uniform(0.3, 0.5)

        positions.append((x, y))
        radii.append(radius)

        circle = Circle((x, y), radius, facecolor=colors[i % len(colors)],
                        alpha=0.6, edgecolor='black', linewidth=1.5,
                        label=f"Set {i + 1}")
        ax.add_patch(circle)
        legend_handles.append(circle)

    ax.legend(handles=legend_handles, loc='upper right', fontsize=12,
              framealpha=0.9, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    return fig, ax


relationship_types = ['intersect', 'contained', 'no_overlap', 'slight_overlap']

# for rel_type in relationship_types:
#     for num_sets in [2, 3, 4]:
#         fig, _ = generate_random_venn(sets=num_sets, relationship_type=rel_type)
#         plt.savefig(f'venn_{rel_type}_{num_sets}sets.png', dpi=150, bbox_inches='tight')
#         plt.close()
