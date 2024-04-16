import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, repulsive_edge_index):
        super(CustomGCNConv, self).__init__(in_channels, out_channels)
        self.repulsive_edge_index = repulsive_edge_index

    def forward(self, x, edge_index):
        # Standard GCN forward pass
        x = super().forward(x, edge_index)
        
        # Logic for repulsive edges
        row, col = self.repulsive_edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return x - norm[:, None] * x[col]

class GCN(torch.nn.Module):
    def __init__(self, dataset, repulsive_edge_index):
        super(GCN, self).__init__()
        self.conv1 = CustomGCNConv(dataset.num_features, 16, repulsive_edge_index)
        self.conv2 = CustomGCNConv(16, dataset.num_classes, repulsive_edge_index)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
