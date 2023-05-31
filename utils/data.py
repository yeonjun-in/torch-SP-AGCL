import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, dense_to_sparse
from deeprobust.graph.data import Dataset, PrePtbDataset
import json
import scipy.sparse as sp
from deeprobust.graph.global_attack import Random

def get_data(root, name, attack, ptb_rate):
    if name in ['cora', 'citeseer', 'pubmed']:
        data = Dataset(root=root, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        dataset = Data()
        
        dataset.x = torch.from_numpy(features.toarray()).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        
        dataset.train_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_train)).bool()
        dataset.val_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_val)).bool()
        dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test)).bool()

        if attack == 'meta':
            if ptb_rate == 0.0:
                return [dataset]
            else:
                perturbed_data = PrePtbDataset(root=root,
                                name=name,
                                attack_method=attack,
                                ptb_rate=ptb_rate)
                perturbed_adj = perturbed_data.adj
                dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()            
                return [dataset]
        elif attack == 'nettack':
            if ptb_rate == 0.0:
                with open(f'{root}{name}_nettacked_nodes.json') as json_file:
                    ptb_idx = json.load(json_file)
                    idx_test_att = ptb_idx['attacked_test_nodes']
                    dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test_att)).bool()
            else:
                perturbed_adj = sp.load_npz(f'{root}{name}_nettack_adj_{int(ptb_rate)}.0.npz')
                with open(f'{root}{name}_nettacked_nodes.json') as json_file:
                    ptb_idx = json.load(json_file)
                
                dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
                idx_test_att = ptb_idx['attacked_test_nodes']
                dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test_att)).bool()
            
            return [dataset]

        elif attack == 'random':
            if ptb_rate == 0.0:
                return [dataset]
            attacker = Random()
            n_perturbations = int(ptb_rate * (dataset.edge_index.shape[1]//2))
            attacker.attack(adj, n_perturbations, type='add')
            perturbed_adj = attacker.modified_adj
            dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()            
            return [dataset]
      
    else:
        if name in ['photo', 'computers', 'cs', 'physics']:
            from torch_geometric.datasets import Planetoid, Amazon, Coauthor
            # if name in ['cora_pyg', 'citeseer_pyg']:
            #     data = Planetoid(root=root, name=name.split('_')[0])[0] 
            if name in ['photo', 'computers']:
                data = Amazon(root=root, name=name)[0]
                if ptb_rate > 0:
                    edge = torch.load(f'{root}{name}_{attack}_adj_{ptb_rate}.pt').cpu()
                    data.edge_index = edge
                else:
                    data.edge_index = to_undirected(data.edge_index)
                data = create_masks(data)
            elif name in ['cs', 'physics']:
                data = Coauthor(root=root, name=name)[0]
                if ptb_rate > 0:
                    edge = torch.load(f'{root}{name}_{attack}_adj_{ptb_rate}.pt').cpu()
                    data.edge_index = edge
                else:
                    data.edge_index = to_undirected(data.edge_index)
                data = create_masks(data)

        elif name in ['squirrel', 'chameleon', 'actor', 'cornell', 'wisconsin', 'texas']:
            from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
            if name in ['squirrel', 'chameleon']:
                data = WikipediaNetwork(root=root, name=name)[0]
                data.edge_index = to_undirected(data.edge_index)
                data = create_masks(data)
            if name in ['actor']:
                data = Actor(root=root)[0]
                data.edge_index = to_undirected(data.edge_index)
                data = create_masks(data)
            if name in ['cornell', 'wisconsin', 'texas']:
                data = WebKB(root=root, name=name)[0]
                data.edge_index = to_undirected(data.edge_index)
                data = create_masks(data)
           
        return [data]


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    tr = 0.1
    vl = 0.1
    tst = 0.8
    if not hasattr(data, "val_mask"):
        _train_mask = _val_mask = _test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * vl)
            test_size = int(labels.shape[0] * tst)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if _train_mask is None:
                _train_mask = train_mask
                _val_mask = dev_mask
                _test_mask = test_mask

            else:
                _train_mask = torch.cat((_train_mask, train_mask), dim=0)
                _val_mask = torch.cat((_val_mask, dev_mask), dim=0)
                _test_mask = torch.cat((_test_mask, test_mask), dim=0)
        
        data.train_mask = _train_mask.squeeze()
        data.val_mask = _val_mask.squeeze()
        data.test_mask = _test_mask.squeeze()
    
    elif hasattr(data, "val_mask") and len(data.val_mask.shape) == 1:
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T
        data.test_mask = data.test_mask.T
    
    else:  
        num_folds = torch.min(torch.tensor(data.train_mask.size())).item()
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T
        if len(data.test_mask.size()) == 1: 
            data.test_mask = data.test_mask.unsqueeze(0).expand(num_folds, -1) 
        else:
            data.test_mask = data.test_mask.T

    return data

