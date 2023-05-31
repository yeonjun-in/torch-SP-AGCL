import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from utils import get_data, set_everything, set_cuda_device
from torch_geometric.utils import to_dense_adj, coalesce, to_undirected
import pandas as pd

'''
photo     6000, 12000, 18000, 24000, 30000
computers 12000, 24000, 36000, 48000, 60000
cs        4000, 8000, 12000, 16000, 20000
physics   12000, 24000, 36000, 48000, 60000
'''

dic = {'photo':30000//300, 'computers':60000//300, 'cs':20000//200, 'physics':60000//200}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='computers', type=str)
parser.add_argument('--ptb_rate', default=0.05, type=float)

args = parser.parse_args()

data_home = f'./dataset/'
data = get_data(data_home, args.dataset, 'meta', 0.0)[0]
adj = to_dense_adj(data.edge_index)[0]
adj_ = adj.clone()
folders = os.listdir('perturbed_graph')
folders = sorted([f for f in folders if args.dataset in f])
idx = int(dic[args.dataset] / (0.25/args.ptb_rate))
adjs = []
for f in folders[:idx]:
    tmp = torch.load(f'perturbed_graph/{f}', map_location='cpu')
    idx2nodes = {i:n for i, n in enumerate(tmp['nodes'])}
    print(f, (tmp['modified'].cpu()>tmp['clean']).sum(), (tmp['modified'].cpu()<tmp['clean']).sum())
    df = pd.DataFrame(tmp['modified'].nonzero().cpu().numpy())
    df[0] = df[0].map(idx2nodes)
    df[1] = df[1].map(idx2nodes)
    adjs.append(torch.from_numpy(df.values.T))
adjs.append(data.edge_index)    
full_adj = torch.cat(adjs, dim=1)
attack_edge = to_undirected(coalesce(full_adj))

print((attack_edge.shape[1]-data.edge_index.shape[1]) / data.edge_index.shape[1]*100)

torch.save(attack_edge, f'{data_home}{args.dataset}_meta_adj_{args.ptb_rate}.pt')