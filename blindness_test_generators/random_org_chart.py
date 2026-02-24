import matplotlib.pyplot as plt
import networkx as nx
import random


def create_random_org_chart():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create tree structure
    num_levels = 4
    max_children_per_node = 4

    G = nx.DiGraph()
    G.add_node(0, level=0)  # Root

    node_counter = 1
    current_level_nodes = [0]

    for level in range(1, num_levels):
        next_level_nodes = []
        for parent in current_level_nodes:
            num_children = random.randint(0, max_children_per_node)
            for _ in range(num_children):
                G.add_node(node_counter, level=level)
                G.add_edge(parent, node_counter)
                next_level_nodes.append(node_counter)
                node_counter += 1
        current_level_nodes = next_level_nodes

    # Create hierarchical layout
    pos = nx.multipartite_layout(G, subset_key="level", align="horizontal")

    # Labels
    labels = {
        # Only include a label if below a certain depth
        node: (
            node
            if G.nodes[node]['level'] < 2
            else ''
        )
        for node in G.nodes()
    }

    # Draw
    nx.draw(G, pos, ax=ax,
            labels=labels,
            node_color='lightblue',
            node_size=2000,
            font_size=9,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20)

    plt.tight_layout()
    plt.show()


create_random_org_chart()