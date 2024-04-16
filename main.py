from torch_geometric.datasets import Planetoid
from repulsive_graph import RepulsiveGraphCreator
from models import GCN

def main():
    # Load data
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # Create repulsive graph
    rep_creator = RepulsiveGraphCreator(data)
    repulsive_edge_index = rep_creator.create_repulsive_graph()
    
    # Initialize model
    model = GCN(dataset, repulsive_edge_index)
    print(model)

if __name__ == '__main__':
    main()
