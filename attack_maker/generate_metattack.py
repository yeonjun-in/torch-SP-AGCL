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
from torch_geometric.utils import to_dense_adj

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=6, type=int)
parser.add_argument('--sub_size', default=3000, type=int)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cs', choices=['cora', 'citeseer', 'photo', 'computers', 'cs', 'physics'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate', choices=[0.05, 0.1, 0.15, 0.2, 0.25])
parser.add_argument('--ptb_n', type=int, default=200)
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self', 'Meta-Train','A-Meta-Train'], help='model variant')

args = parser.parse_args()
set_everything(args.seed)

save_dict = {}

set_cuda_device(args.device)
device = f'cuda:{args.device}'
data_home = f'./dataset/'
data = get_data(data_home, args.dataset, 'meta', 0.0)[0]
adj_np = to_dense_adj(data.edge_index)[0].numpy().astype(np.float32)
adj = sp.csr_matrix(adj_np)

features = data.x.numpy().astype(np.float32)
features = sp.csr_matrix(features)
labels = data.y.numpy()
idx_train = data.train_mask[0, :].nonzero().flatten().numpy()
idx_val = data.val_mask[0, :].nonzero().flatten().numpy()
idx_test = data.test_mask[0, :].nonzero().flatten().numpy()
idx_unlabeled = np.union1d(idx_val, idx_test)

# for seed in range(int(args.ptb_rate*100)):
    
nodes = torch.randperm(data.x.size(0))[:args.sub_size].sort()[0].numpy()
save_dict['nodes'] = nodes
sub_adj = adj[nodes, :][:, nodes]
sub_x = features[nodes, :]
sub_y = labels[nodes]

sub_idx_train = np.sort(np.in1d(nodes, idx_train).nonzero()[0])
sub_idx_val = np.sort(np.in1d(nodes, idx_val).nonzero()[0])
sub_idx_test = np.sort(np.in1d(nodes, idx_test).nonzero()[0])
sub_idx_unlabeled = np.sort(np.in1d(nodes, idx_unlabeled).nonzero()[0])

perturbations = args.ptb_n ##int(0.01 * (sub_adj.sum()//2))
sub_adj, sub_x, sub_y = preprocess(sub_adj, sub_x, sub_y, preprocess_adj=False)

save_dict['clean'] = sub_adj
# Setup Surrogate Model
surrogate = GCN(nfeat=sub_x.shape[1], nclass=sub_y.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(sub_x, sub_adj, sub_y, sub_idx_train)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=sub_adj.shape[0], feature_shape=sub_x.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=sub_adj.shape[0], feature_shape=sub_x.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=sub_x.shape[1],
              nhid=args.hidden,
              nclass=sub_y.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(sub_x, sub_adj, sub_y, sub_idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[sub_idx_test], sub_y[sub_idx_test])
    acc_test = accuracy(output[sub_idx_test], sub_y[sub_idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    model.attack(sub_x, sub_adj, sub_y, sub_idx_train, sub_idx_unlabeled, perturbations, ll_constraint=False)
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    modified_adj = model.modified_adj
    save_dict['modified'] = modified_adj
    # modified_features = model.modified_features
    test(modified_adj)

    # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./perturbed_graph', name='{}_meta_adj_{}_{}'.format(args.dataset, args.ptb_rate, args.seed))
    # torch.save(nodes, f'./perturbed_graph/{args.dataset}_meta_{args.ptb_rate}_subnodes_{args.seed}.pt')
    # model.save_features(root='./', name='mod_features')
    torch.save(save_dict, f'./perturbed_graph/{args.dataset}_meta_seed{args.seed}.pt')
if __name__ == '__main__':
    main()
