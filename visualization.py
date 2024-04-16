import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(edge_index, node_labels, title="Graph Visualization"):
    G = nx.Graph()
    num_nodes = len(node_labels)

    # Add nodes with their labels for coloring
    for node in range(num_nodes):
        G.add_node(node, label=node_labels[node])

    # Add edges
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

    # Define color mapping
    color_map = [plt.get_cmap('viridis')(label) for label in node_labels]

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, node_color=color_map, with_labels=True, node_size=70, font_size=10)
    plt.title(title)
    plt.show()
