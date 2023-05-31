import torch
import random, os
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import to_dense_adj, add_self_loops

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def dense_to_sparse_adj(edge_index, n_node):
    return torch.sparse.FloatTensor(edge_index,
                                    torch.ones(edge_index.shape[1]).to(edge_index.device),
                                    [n_node, n_node])

def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index,
                                    torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])

def to_dense_subadj(edge_index, subsize):
    edge = add_self_loops(edge_index, num_nodes=subsize)[0]
    return to_dense_adj(edge)[0].fill_diagonal_(0.0)

def set_cuda_device(device_num):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device) 

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        
        if name not in ['device', 'patience', 'epochs', 'save_dir', 'in_dim', 'n_class', 'best_epoch', 'save_fig', 'n_node', 'n_degree', 'attack', 'attack_type', 'ptb_rate', 'verbose', 'mm', '']:
            st_ = "{}:{} / ".format(name, val)
            st += st_
        
    
    return st[:-1]

def set_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # specify GPUs locally

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
