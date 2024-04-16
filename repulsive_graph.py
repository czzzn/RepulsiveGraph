import torch

class RepulsiveGraphCreator:
    def __init__(self, data):
        self.data = data
    
    def create_repulsive_graph(self):
        # Define your own logic here for creating Gr
        edge_index = self.data.edge_index
        num_nodes = self.data.num_nodes
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
        adjacency[edge_index[0], edge_index[1]] = 1
        repulsive_edges = 1 - adjacency - torch.eye(num_nodes, dtype=torch.int)
        repulsive_edge_index = repulsive_edges.nonzero(as_tuple=False).t()
        return repulsive_edge_index
