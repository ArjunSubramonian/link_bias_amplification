from typing import Optional, Callable

import numpy as np
import torch
import pandas as pd

from torch_geometric.data import Data, InMemoryDataset

class DBLPFairness(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'dblp_fairness.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long).contiguous()

        data = Data(x=x[:, 1:], y=y, edge_index=edge_index, sens=x[:, 0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
