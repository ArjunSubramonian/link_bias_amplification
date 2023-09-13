# +
# Adapted from FairGNN repository (https://github.com/EnyanDai/FairGNN)
# -

from typing import Optional, List, Callable

import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os.path as osp

from torch_geometric.data import Data, InMemoryDataset

class SayNo(InMemoryDataset):
    
    url = 'https://github.com/EnyanDai/FairGNN/blob/main/dataset'
    
    def __init__(self, root: str, name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        self.mode = "raw"
        
        self.dataset_name = "region_job"
        self.sens_attr = "region"
        self.predict_attr = "I_am_working_in_field"
        if self.name == 'Pokec-n':
            self.dataset_name = "region_job_2"
        elif self.name == 'NBA':
            self.dataset_name = 'nba'
            self.sens_attr = "country"
            self.predict_attr = "AGE"
            
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
        names = ['.csv', '.embedding', '_relationship.txt']
        return [f'{self.dataset_name.lower()}{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def _process_helper(self):
        dataset = self.dataset_name
        sens_attr = self.sens_attr
        predict_attr = self.predict_attr
        
        # load data
        print('Loading {} dataset'.format(dataset))

        idx_features_labels = pd.read_csv(self.raw_paths[0])
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = (idx_features_labels[predict_attr].values > 25).astype(float)

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(self.raw_paths[2], dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), 
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        sens = idx_features_labels[sens_attr].values
        sens = torch.LongTensor(sens)

        return adj, features, labels, sens

    def _process_emb_helper(self):
        dataset = self.dataset_name
        sens_attr = self.sens_attr
        predict_attr = self.predict_attr
        
        print('Loading {} dataset'.format(dataset))

        graph_embedding = np.genfromtxt(
            self.raw_paths[1],
            skip_header=1,
            dtype=float)
        embedding_df = pd.DataFrame(graph_embedding)
        embedding_df[0] = embedding_df[0].astype(int)
        embedding_df = embedding_df.rename(index=int, columns={0: "user_id"})

        idx_features_labels = pd.read_csv(self.raw_paths[0])
        idx_features_labels = pd.merge(idx_features_labels, embedding_df, how="left", on="user_id")
        idx_features_labels = idx_features_labels.fillna(0)

        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = (idx_features_labels[predict_attr].values > 25).astype(float)

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(self.raw_paths[2], dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                            dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        sens = idx_features_labels[sens_attr].values
        sens = torch.LongTensor(sens)

        return adj, features, labels, sens
    
    def process(self):
        if self.mode == 'emb':
            adj, x, y, sens = self._process_emb_helper()
        else:
            adj, x, y, sens = self._process_helper()
        
        adj = adj.tocoo()
        row, col = adj.row, adj.col
        edge_index = torch.tensor([row, col]).long()

        data = Data(x=x, y=y, edge_index=edge_index, sens=sens)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
