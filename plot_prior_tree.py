import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the priors from the pickle file
with open('priors.pkl', 'rb') as f:
    priors = pickle.load(f)


def plot_tree(t, ax):

    # priors_t has shape corresponding to: first dim => edges to nodes 1 & 2,
    # second dim => edges to nodes 3..6, third dim => edges to nodes 7..14.
    priors_t = priors[t].sum(axis=(0, 2, 4))
    priors_t_stage0 = priors_t.sum(axis=(1, 2))
    priors_t_stage1 = priors_t.sum(axis=(0, 2))
    priors_t_stage2 = priors_t.sum(axis=(0, 1))

    color_values = np.concatenate((priors_t_stage0, priors_t_stage1, priors_t_stage2))

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (0..14) and edges for a four-stage binary decision tree
    nodes = [str(i) for i in range(15)]
    edges = [
        ('0', '1'), ('0', '2'),      # first dim (2 edges)
        ('1', '3'), ('1', '4'),      # second dim (4 edges) 
        ('2', '5'), ('2', '6'),
        ('3', '7'), ('3', '8'),      # third dim (8 edges)
        ('4', '9'), ('4', '10'),
        ('5', '11'), ('5', '12'),
        ('6', '13'), ('6', '14')
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Define positions for each node
    pos = {
        '0': (0, 3),
        '1': (-2, 2), '2': (2, 2),
        '3': (-3, 1), '4': (-1, 1), '5': (1, 1), '6': (3, 1),
        '7': (-3.5, 0), '8': (-2.5, 0), '9': (-1.5, 0), '10': (-0.5, 0),
        '11': (0.5, 0), '12': (1.5, 0), '13': (2.5, 0), '14': (3.5, 0)
    }

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=100, ax=ax)

    # Determine edge colors
    cmap = plt.cm.inferno
    edge_colors = [cmap(val) for val in color_values]

    # Determine edge widths
    min_width = 2
    max_width = 10
    edge_widths = min_width + (max_width - min_width) * color_values

    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, ax=ax)

    # Draw labels
    #nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)

    ax.set_title(f't = {t}')
    ax.axis('off')

# Create figure with 4 subplots - trees and colorbar
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 4, width_ratios=[4, 4, 4, 0.3])

# Create axes for trees and colorbar
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1]) 
ax3 = fig.add_subplot(gs[2])
cax = fig.add_subplot(gs[3])

# Plot three trees
plot_tree(0, ax1)
plot_tree(5, ax2)
plot_tree(50, ax3)

# Create colorbar
norm = plt.Normalize(0, 1)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno), 
                   cax=cax,
                   label='Probability')

plt.tight_layout()
plt.show()

a = 1