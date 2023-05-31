import torch
from torch.optim import AdamW
from embedder import embedder
from encoder import GCN, GCNLayer
from utils import get_graph_drop_transform, set_everything, dense_to_sparse_x, to_dense_subadj
from copy import deepcopy
from collections import defaultdict
from torch_geometric.utils import to_undirected, to_dense_adj, dense_to_sparse, subgraph, add_self_loops
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.utils import to_dense_subadj

class SPAGCL_node(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
    
    def attack_adj(self, x1, x2, n_edge):
        n_nodes = len(x1.x)
        add_edge_num = int(self.args.add_edge_rate * n_edge)
        drop_edge_num = int(self.args.drop_edge_rate * n_edge)
        grad_sum = x1.edge_adj.grad + x2.edge_adj.grad
        grad_sum_1d = grad_sum.view(-1)
        values, indices = grad_sum_1d.sort()
        add_idx, drop_idx = indices[-add_edge_num:], indices[:drop_edge_num]
        
        add_idx_dense = torch.stack([add_idx // n_nodes, add_idx % n_nodes])
        drop_idx_dense = torch.stack([drop_idx // n_nodes, drop_idx % n_nodes])
        
        add = to_dense_adj(add_idx_dense, max_num_nodes=n_nodes)[0]
        drop = to_dense_adj(drop_idx_dense, max_num_nodes=n_nodes)[0]
        return add, 1-drop
    
    def attack_feat(self, x1, x2):
        n_nodes, n_dim = x1.x.size()
        drop_feat_num = int((n_dim * self.args.drop_feat_rate) * n_nodes)
        grad_sum = x1.x.grad + x2.x.grad
        grad_sum_1d = grad_sum.view(-1)
        values, indices = grad_sum_1d.sort()
        
        drop_idx = indices[:drop_feat_num]
        
        drop_idx_dense = torch.stack([drop_idx // n_dim, drop_idx % n_dim])
        
        drop_sparse = dense_to_sparse_x(drop_idx_dense, n_nodes, n_dim)
        return 1-drop_sparse.to_dense()

    def subgraph_sampling(self, data1, data2):
        self.sample_size = min(self.args.sub_size, self.args.n_node)
        nodes = torch.randperm(data1.x.size(0))[:self.sample_size].sort()[0]
        edge1, edge2 = add_self_loops(data1.edge_index, num_nodes=data1.x.size(0))[0], add_self_loops(data2.edge_index, num_nodes=data1.x.size(0))[0]
        edge1 = subgraph(subset=nodes, edge_index=edge1, relabel_nodes=True)[0]
        edge2 = subgraph(subset=nodes, edge_index=edge2, relabel_nodes=True)[0]
        
        tmp1, tmp2 = Data(), Data()
        tmp1.x, tmp2.x = data1.x[nodes], data2.x[nodes]
        tmp1.edge_index, tmp2.edge_index = edge1, edge2

        return tmp1, tmp2
        
    def training(self):
        
        self.train_result, self.val_result, self.test_result = defaultdict(list), defaultdict(list), defaultdict(list)
        for seed in range(self.args.seed_n):
            self.seed = seed
            set_everything(seed)
            
            data = self.data.clone()
            
            knn_data = Data()
            sim = F.normalize(data.x).mm(F.normalize(data.x).T).fill_diagonal_(0.0)       
            dst = sim.topk(self.args.knn, 1)[1]
            src = torch.arange(data.x.size(0)).unsqueeze(1).expand_as(sim.topk(self.args.knn, 1)[1])
            edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)])
            edge_index = to_undirected(edge_index)
            knn_data.x = deepcopy(data.x)
            knn_data.edge_index = edge_index
            data = data.cuda()
            knn_data = knn_data.cuda()

            data.edge_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0].to_sparse()
            
            transform_1 = get_graph_drop_transform(drop_edge_p=self.args.d_1, drop_feat_p=self.args.d_1)
            transform_2 = get_graph_drop_transform(drop_edge_p=self.args.d_2, drop_feat_p=self.args.d_2)
            transform_3 = get_graph_drop_transform(drop_edge_p=self.args.d_3, drop_feat_p=self.args.d_3)

            self.encoder = GCN(GCNLayer, [self.args.in_dim] + self.args.layers, batchnorm=self.args.bn)   # 512, 256, 128
            self.model = modeler(self.encoder, self.args.layers[-1], self.args.layers[-1], self.args.tau).cuda()
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

            best, cnt_wait = 0, 0
            for epoch in range(1, self.args.epochs+1):

                sub1, sub2 = self.subgraph_sampling(data, knn_data)
                self.model.train()
                self.optimizer.zero_grad()

                x1, x2, x_knn, x_adv = transform_1(sub1), transform_2(sub1), transform_3(sub2), deepcopy(sub1)
                x1.edge_adj, x2.edge_adj = to_dense_subadj(x1.edge_index, self.sample_size), to_dense_subadj(x2.edge_index, self.sample_size)
                x_knn.edge_adj, x_adv.edge_adj = to_dense_subadj(x_knn.edge_index, self.sample_size), to_dense_subadj(x_adv.edge_index, self.sample_size)
                
                if epoch > self.args.warmup:
                    x1.edge_adj = x1.edge_adj.requires_grad_()
                    x2.edge_adj = x2.edge_adj.requires_grad_()
                    
                    x1.x = x1.x.requires_grad_()
                    x2.x = x2.x.requires_grad_()
                
                z1 = self.model(x1.x, x1.edge_adj)
                z2 = self.model(x2.x, x2.edge_adj)
                loss = self.model.loss(z1, z2, batch_size=0)
                
                loss.backward()
        
                if epoch > self.args.warmup:
                    n_edge = int(x1.edge_adj.sum().item())
                    add_edge, masking_edge = self.attack_adj(x1, x2, n_edge=n_edge)
                    masking_feat = self.attack_feat(x1, x2)
                    
                    x1.edge_adj, x2.edge_adj = x1.edge_adj.detach(), x2.edge_adj.detach()
                    x_adv.edge_adj = ((x1.edge_adj*masking_edge) + add_edge*1.0).clamp(0, 1).detach()

                    x1.x, x2.x = x1.x.detach(), x2.x.detach()
                    x_adv.x = (x1.x*masking_feat).detach()
                    
                    x_knn.x = x_knn.x.detach()
                    x_knn.edge_adj = x_knn.edge_adj.detach()

                    z1 = self.model(x1.x, x1.edge_adj.to_sparse())
                    z2 = self.model(x2.x, x2.edge_adj.to_sparse())
                    z_adv = self.model(x_adv.x, x_adv.edge_adj.to_sparse())
                    z_knn = self.model(x_knn.x, x_knn.edge_adj.to_sparse())
                    loss = self.model.loss(z1, z2, batch_size=0)*0.5
                    loss += self.model.loss(z1, z_adv, batch_size=0)*self.args.lambda_1*0.5
                    loss += self.model.loss(z1, z_knn, batch_size=0)*self.args.lambda_2*0.5
                    # print(self.args.lambda_1*0.5, self.args.lambda_2*0.5)
                    self.optimizer.zero_grad()
                    loss.backward()
                
                self.optimizer.step()

                print(f'Epoch {epoch}: Loss {loss.item()}')

                if epoch % self.args.verbose == 0:
                    val_acc = self.verbose(data)
                    if val_acc > best:
                        best = val_acc
                        cnt_wait = 0
                        torch.save(self.model.online_encoder.state_dict(), '{}/saved_model/best_{}_{}_{}_{}_{}_seed{}.pkl'.format(self.args.save_dir, self.args.dataset, self.args.attack, self.args.ptb_rate, self.args.attack_type, self.args.embedder, seed))
                    else:
                        cnt_wait += self.args.verbose
                    
                    if cnt_wait == self.args.patience:
                        print('Early stopping!')
                        break
            
            self.model.online_encoder.load_state_dict(torch.load('{}/saved_model/best_{}_{}_{}_{}_{}_seed{}.pkl'.format(self.args.save_dir, self.args.dataset, self.args.attack, self.args.ptb_rate, self.args.attack_type, self.args.embedder, seed), map_location=f'cuda:{self.args.device}'))
            if self.args.save_embed:
                self.get_embeddings(data)
            only_clean = True if self.args.dataset in ['squirrel', 'chameleon', 'texas', 'wisconsin', 'cornell', 'actor'] else False
            if self.args.task == 'node':
                if self.args.attack_type == 'evasive':
                    self.eval_clean_and_evasive(data, only_clean)
                elif self.args.attack_type == 'poison':
                    self.eval_poisoning(data)
                
        self.summary_result()

        
class modeler(torch.nn.Module):
    def __init__(self, encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(modeler, self).__init__()
        self.online_encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
