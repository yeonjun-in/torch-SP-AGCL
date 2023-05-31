import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from collections import defaultdict
from utils import get_data, to_numpy, config2string
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
    
parser = argparse.ArgumentParser()

parser.add_argument('--embedder', default='ariel')
parser.add_argument('--dataset', default='pubmed')
parser.add_argument('--task', default='clustering')
parser.add_argument('--attack', default='meta')
parser.add_argument('--attack_type', default='poison')
parser.add_argument('--ptb_rate', default=0.25, type=float)
parser.add_argument('--save_dir', type=str, default='./results')

args, _ = parser.parse_known_args()


def clustering(embed, data):
    embed = torch.load(f'results/embeddings/{args.embedder}/{args.dataset}_{args.attack_type}_meta_{args.ptb_rate}_embed_seed{seed}.pt')
    embed = F.normalize(embed, dim=1, p=2)
    embed = to_numpy(embed)
    y = to_numpy(data.y)
    cluster = KMeans(n_clusters=data.y.unique().size(0), random_state=1995).fit(embed).labels_
    nmi = normalized_mutual_info_score(cluster, y)

    return nmi

def summary_result():
    config_str = config2string(args)
    assert train_result.keys() == val_result.keys() and train_result.keys() == test_result.keys()

    key = list(train_result.keys())

    result_path = f'{args.save_dir}/summary_result/{args.embedder}/{args.embedder}_{args.dataset}_{args.task}_{args.attack}_{args.attack_type}.txt'
    mode = 'a' if os.path.exists(result_path) else 'w'
    with open(result_path, mode) as f:
        f.write(config_str)
        f.write(f'\n')
        for k in key:
            f.write(f'====={k}=====')
            f.write(f'\n')
            train_mean, train_std = np.mean(train_result[k]), np.std(train_result[k])
            val_mean, val_std = np.mean(val_result[k]), np.std(val_result[k])
            test_mean, test_std = np.mean(test_result[k]), np.std(test_result[k])
            f.write(f'Train NMI: {train_mean*100:.2f}±{train_std*100:.2f}, Val NMI: {val_mean*100:.2f}±{val_std*100:.2f}, Test NMI: {test_mean*100:.2f}±{test_std*100:.2f}')
            f.write(f'\n')
            if args.attack_type == 'poison' and args.ptb_rate == 0.25:
                f.write(f'='*40)
                f.write(f'\n')
            

if args.ptb_rate==0:
    args.attack_type='evasive'
else:
    args.attack_type='poison'

print(args.dataset)

data = get_data('./dataset/', args.dataset, args.attack, args.ptb_rate)[0]
train_result, val_result, test_result = defaultdict(list), defaultdict(list), defaultdict(list)
for seed in range(10):
    try:
        embed = torch.load(f'results/embeddings/{args.embedder}/{args.dataset}_{args.attack_type}_meta_{args.ptb_rate}_embed_seed{seed}.pt')
    except:
        continue
    nmi = clustering(embed, data)
    train_result[f'POISON_{args.attack}_{args.ptb_rate}'].append(nmi); val_result[f'POISON_{args.attack}_{args.ptb_rate}'].append(nmi); test_result[f'POISON_{args.attack}_{args.ptb_rate}'].append(nmi)
    
summary_result()

    
