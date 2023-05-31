import torch
import argparse
from utils import set_everything
import warnings 
warnings.filterwarnings("ignore")

def parse_args():
    # input arguments
    set_everything(1995)
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedder', default='SPAGCL_node')
    parser.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics', 'chameleon', 'squirrel', 'actor', 'texas', 'wisconsin', 'cornell'])
    parser.add_argument('--task', default='node', choices=['clustering', 'node', 'link'])
    parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'nettack', 'random', 'feat_gau', 'feat_bern'])
    parser.add_argument('--attack_type', type=str, default='poison', choices=['poison', 'evasive'])
    if parser.parse_known_args()[0].attack_type in ['poison']:
        parser.add_argument('--ptb_rate', type=float, default=0.0)
    
    parser.add_argument('--seed_n', default=3, type=int)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument("--layers", nargs='*', type=int, default=[512, 128], help="The number of units of each layer of the GNN. Default is [256]")
    
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--wd', type=float, default=1e-5)

    parser.add_argument("--save_embed", action='store_true', default=False)
    
    parser.add_argument('--lambda_1', type=float, default=2.0)
    parser.add_argument('--lambda_2', type=float, default=2.0)
    
    parser.add_argument('--d_1', type=float, default=0.3)
    parser.add_argument('--d_2', type=float, default=0.2)
    parser.add_argument('--d_3', type=float, default=0.0)
    parser.add_argument("--bn", action='store_false', default=True)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--sub_size', type=int, default=5000)
    parser.add_argument('--add_edge_rate', type=float, default=0.3)
    parser.add_argument('--drop_feat_rate', type=float, default=0.3)
    parser.add_argument('--knn', type=int, default=10)
    
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--patience', type=int, default=400)
    parser.add_argument('--verbose', type=int, default=10)
    
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--save_fig', action='store_true', default=True)

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    args.drop_edge_rate = args.add_edge_rate

    assert ~(args.attack_type == 'poison' and args.ptb_rate == 0.0)
    if args.attack_type == 'evasive':
        args.ptb_rate = 0.0
    if '_link' in args.embedder:
        args.task = 'link'

    torch.cuda.set_device(args.device)

    if args.embedder == 'SPAGCL_node':
        from models import SPAGCL_node
        embedder = SPAGCL_node(args)
    if args.embedder == 'SPAGCL_link':
        from models import SPAGCL_link
        embedder = SPAGCL_link(args)

    embedder.training()

if __name__ == '__main__':
    main()
