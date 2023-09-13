import torch
import copy

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix

class ConnectedClasses(BaseTransform):
    
    def __init__(self, connection: str = 'weak'):
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        self.connection = connection

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(
            adj, connection=self.connection)

        component = list(component)
        label = data.y.tolist()
        pair = list(zip(label, component))
        uniq_pair = sorted(set(pair))
        relabel_map = {v : k for k, v in enumerate(uniq_pair)}
        relabel = [relabel_map[e] for e in pair]
        
        new_data = copy.copy(data)
        new_data.y = torch.tensor(relabel)

        return new_data
   
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class LargestConnectedComponents(BaseTransform):
    def __init__(self, num_components: int = 1, connection: str = 'weak'):
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        self.num_components = num_components
        self.connection = connection

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(
            adj, connection=self.connection)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'

class LargestBiconnectedComponents(BaseTransform):
    def __init__(self, num_components: int = 1, connection: str = 'weak'):
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        self.num_components = num_components
        self.connection = connection

    def __call__(self, data: Data) -> Data:
        import networkx as nx
        from torch_geometric.utils import to_networkx

        G = to_networkx(data, to_undirected=True)
        components = list(nx.biconnected_components(G))
        components.sort(key=len, reverse=True)
        for c in components:
            if len(c) == 2:
                break
            print(len(c))
        largest_component = torch.tensor(list(components[0]))
        return data.subgraph(largest_component)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'
