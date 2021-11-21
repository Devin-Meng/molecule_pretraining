import torch
import numpy as np
import sys
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ogb.utils import smiles2graph

import pyximport
sys.path.append('/home/Zhaoxu/molecule_pretraining/src/method/data')
pyximport.install(pyximport=True, setup_args={'include_dirs': np.get_include()})
import algos

class MolDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        smi = self.data[idx]['smi']
        charges = self.data[idx]['charges']
        graph = smiles2graph(smi)

        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        data = Data()
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        data.y = torch.Tensor(charges)
        data.idx = idx
        
        return preprocess_item(data)

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.N = x.size(0)

    return item
