# +
# MIT License

# Copyright (c) 2021 Chirag Agarwal

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -

from typing import Optional, List, Callable

import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os.path as osp

from torch_geometric.data import Data, InMemoryDataset

class Nifty(InMemoryDataset):
    
    url = 'https://github.com/chirag126/nifty/tree/main/dataset'
    
    def __init__(self, root: str, name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        
        if self.name == 'Credit':
            self.sens_attr = "Age"
            self.predict_attr = "EducationLevel"
        elif self.name == 'German':
            self.sens_attr = "Gender"
            self.predict_attr = "ForeignWorker"
            
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
        
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
        
    @property
    def raw_file_names(self) -> List[str]:
        names = ['.csv', '_edges.txt']
        return [f'{self.name.lower()}{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def _process_helper(self):
        dataset = self.name
        sens_attr = self.sens_attr
        predict_attr = self.predict_attr
        
        print('Loading {} dataset'.format(dataset))
        
        idx_features_labels = pd.read_csv(self.raw_paths[0])
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        
        if self.name == 'Credit': 
            header.remove('Single')
        elif self.name == 'German':
            header.remove('OtherLoansAtStore')
            header.remove('PurposeOfLoan')
            
            idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
            idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
        
        edges_unordered = np.genfromtxt(self.raw_paths[1]).astype('int')

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        if self.name == 'Credit':
            labels = (idx_features_labels[predict_attr].values > 1).astype(float)
        elif self.name == 'German':
            labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)

        return adj, features, labels, sens
    
    def process(self):
        adj, x, y, sens = self._process_helper()
        
        adj = adj.tocoo()
        row, col = adj.row, adj.col
        edge_index = torch.tensor([row, col]).long()

        data = Data(x=x, y=y, edge_index=edge_index, sens=sens)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
