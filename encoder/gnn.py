import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, LayerNorm, Sequential
from deeprobust.graph import utils
import math

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        adj_norm = adj
        adj_norm = utils.normalize_adj_tensor(adj, sparse=adj.is_sparse)
        support = torch.mm(input, self.weight)
        if adj_norm.is_sparse:
            output = torch.spmm(adj_norm, support)
        else:
            output = torch.mm(adj_norm, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, gcn_module, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=False, weight_standardization=False):
        super().__init__()

        # assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.gcn_module = gcn_module
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        total_layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers = []
            layers.append((self.gcn_module(in_dim, out_dim), 'x, edge_index -> x'),)

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            # else:
            #     layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())
            total_layers.append(Sequential('x, edge_index', layers))
        
        self.model = nn.ModuleList(total_layers)
        # self.model = Sequential('x, edge_index, perturb', layers)

    def forward(self, x, adj, perturb=None):
        if self.weight_standardization:
            self.standardize_weights()
            
        for i, layer in enumerate(self.model):
            x = layer(x, adj)
            if perturb is not None and i==0:
                x += perturb
        return x

    def reset_parameters(self):
        for m in self.model:
            m.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, self.gcn_module):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight