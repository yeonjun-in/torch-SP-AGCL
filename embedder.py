import numpy as np
from utils import get_data, set_cuda_device, config2string, ensure_dir, to_numpy, dense_to_sparse_adj
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class embedder:
    def __init__(self, args):
        print('===',args.dataset, '===') 
        
        self.args = args
        self.device = f'cuda:{args.device}'
        set_cuda_device(args.device)

        self.data_home = f'./dataset/'
        self.data = get_data(self.data_home, args.dataset, args.attack, args.ptb_rate)[0]

        # save results
        self.config_str = config2string(args)
        self.result_path = f'{args.save_dir}/summary_results/{args.dataset}/{args.embedder}.txt'
        
        # basic statistics
        self.args.in_dim = self.data.x.shape[1]
        self.args.n_class = self.data.y.unique().size(0)
        self.args.n_node = self.data.x.shape[0]
        self.embed_dim = args.layers[-1]

        # save path check
        ensure_dir(f'{args.save_dir}/fig/{args.embedder}/{args.dataset}/')
        ensure_dir(f'{args.save_dir}/saved_model/')
        ensure_dir(f'{args.save_dir}/embeddings/{args.embedder}/')
        ensure_dir(f'{args.save_dir}/summary_result/{args.embedder}/bypass/')

    def get_embeddings(self, data):
        tmp_encoder = deepcopy(self.model.online_encoder).eval()
        with torch.no_grad():
            embed = tmp_encoder(data.x, data.edge_adj)
        torch.save(embed.cpu(), f'{self.args.save_dir}/embeddings/{self.args.embedder}/{self.args.dataset}_{self.args.attack_type}_{self.args.attack}_{self.args.ptb_rate}_embed_seed{self.seed}.pt')

    def eval_base(self, data):
        tmp_encoder = deepcopy(self.model.online_encoder).eval()
        with torch.no_grad():
            embed = tmp_encoder(data.x, data.edge_adj)
        embed = F.normalize(embed, dim=1, p=2)
        embed = to_numpy(embed)
        y = to_numpy(data.y)
        
        if len(data.train_mask.size()) == 2:
            train_mask, val_mask, test_mask = to_numpy(data.train_mask[self.seed, :]), to_numpy(data.val_mask[self.seed, :]), to_numpy(data.test_mask[self.seed, :])
        else:
            train_mask, val_mask, test_mask = to_numpy(data.train_mask), to_numpy(data.val_mask), to_numpy(data.test_mask)
        
        linear_model = LogisticRegression(solver='liblinear', multi_class='auto', class_weight=None)
        linear_model.fit(embed[train_mask], y[train_mask])
        pred = linear_model.predict(embed)
        return pred, y, train_mask, val_mask, test_mask
    
    def verbose(self, data):
        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        train_acc = np.mean(correct[train_mask])
        val_acc = np.mean(correct[val_mask])
        test_acc = np.mean(correct[test_mask])

        print(f'====== Train acc {train_acc*100:.2f}, Val acc {val_acc*100:.2f}, Test acc {test_acc*100:.2f},')
        return val_acc
    
    def verbose_link(self, data):

        tmp_encoder = deepcopy(self.model.online_encoder).eval()
        with torch.no_grad():
            embed = tmp_encoder(data.x, data.edge_adj)
        embed = F.normalize(embed, dim=1, p=2)
        embed = to_numpy(embed)
        
        src, dst = data.train_edge_index.cpu().numpy()
        link_emb = np.hstack((embed[src], embed[dst]))
        y = data.train_label.cpu().numpy()
        linear_model = LogisticRegression(solver='lbfgs')
        linear_model.fit(link_emb, y)
        pred = linear_model.predict_proba(link_emb)[:, 1]
        train_auc = roc_auc_score(y, pred)
        
        src, dst = data.val_edge_index.cpu().numpy()
        link_emb = np.hstack((embed[src], embed[dst]))
        y = data.val_label.cpu().numpy()
        pred = linear_model.predict_proba(link_emb)[:, 1]
        val_auc = roc_auc_score(y, pred)
        
        src, dst = data.test_edge_index.cpu().numpy()
        link_emb = np.hstack((embed[src], embed[dst]))
        y = data.test_label.cpu().numpy()
        pred = linear_model.predict_proba(link_emb)[:, 1]
        test_auc = roc_auc_score(y, pred)
        
        print(f'====== Train AUC {train_auc*100:.2f}, Val AUC {val_auc*100:.2f}, Test AUC {test_auc*100:.2f},')

        return train_auc, val_auc, test_auc
    
    def eval_link(self, data):
        save_dict = {'config':self.config_str}
        
        train_auc, val_auc, test_auc = self.verbose_link(data)
        self.train_result[f'Link_{self.args.attack}_{self.args.ptb_rate}'].append(train_auc)
        self.val_result[f'Link_{self.args.attack}_{self.args.ptb_rate}'].append(val_auc)
        self.test_result[f'Link_{self.args.attack}_{self.args.ptb_rate}'].append(test_auc)
    
    def eval_clean_and_evasive(self, data, only_clean=False):
        save_dict = {'config':self.config_str}
        
        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        self.train_result['CLEAN'].append(np.mean(correct[train_mask]))
        self.val_result['CLEAN'].append(np.mean(correct[val_mask]))
        self.test_result['CLEAN'].append(np.mean(correct[test_mask]))
        save_dict['clean'] = [pred, y, train_mask, val_mask, test_mask]
        
        if not only_clean:
            iterator = [0.05, 0.1, 0.15, 0.2, 0.25] if self.args.attack == 'meta' else [0.2, 0.4, 0.6, 0.8, 1.0] if self.args.attack == 'random' else [0.2, 0.4, 0.6, 0.8] if self.args.attack == 'feat_bern' else [0.5, 1, 1.5] if self.args.attack == 'feat_gau' else [1, 2, 3, 4, 5] 
            for ptb in iterator:
                data = get_data(self.data_home, self.args.dataset, self.args.attack, ptb)[0]
                data = data.cuda()
                data.edge_adj = dense_to_sparse_adj(data.edge_index, data.x.size(0))
                
                pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
                correct = (pred==y)
                self.train_result[f'EVASIVE_{self.args.attack}_{ptb}'].append(np.mean(correct[train_mask]))
                self.val_result[f'EVASIVE_{self.args.attack}_{ptb}'].append(np.mean(correct[val_mask]))
                self.test_result[f'EVASIVE_{self.args.attack}_{ptb}'].append(np.mean(correct[test_mask]))
                save_dict[f'evasive_{self.args.attack}_{ptb}'] = [pred, y, train_mask, val_mask, test_mask]

        torch.save(save_dict, f'{self.args.save_dir}/summary_result/{self.args.embedder}/bypass/{self.args.dataset}_{self.args.attack_type}_{self.args.attack}_save_dict_evasive_seed{self.seed}.pt')

    def eval_poisoning(self, data):
        save_dict = {'config':self.config_str}
        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        train_acc = np.mean(correct[train_mask])
        val_acc = np.mean(correct[val_mask])
        test_acc = np.mean(correct[test_mask])

        self.train_result[f'POISON_{self.args.attack}_{self.args.ptb_rate}'].append(np.mean(train_acc))
        self.val_result[f'POISON_{self.args.attack}_{self.args.ptb_rate}'].append(np.mean(val_acc))
        self.test_result[f'POISON_{self.args.attack}_{self.args.ptb_rate}'].append(np.mean(test_acc))
        save_dict[f'poison_{self.args.attack}_{self.args.ptb_rate}'] = [pred, y, train_mask, val_mask, test_mask]
        torch.save(save_dict, f'{self.args.save_dir}/summary_result/{self.args.embedder}/bypass/{self.args.dataset}_{self.args.attack_type}_{self.args.attack}_{self.args.ptb_rate}_save_dict_poison_seed{self.seed}.pt')

    def summary_result(self):
        
        assert self.train_result.keys() == self.val_result.keys() and self.train_result.keys() == self.test_result.keys()

        key = list(self.train_result.keys())

        result_path = f'{self.args.save_dir}/summary_result/{self.args.embedder}/{self.args.embedder}_{self.args.dataset}_{self.args.task}_{self.args.attack}_{self.args.attack_type}.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        with open(result_path, mode) as f:
            f.write(self.config_str)
            f.write(f'\n')
            for k in key:
                f.write(f'====={k}=====')
                f.write(f'\n')
                train_mean, train_std = np.mean(self.train_result[k]), np.std(self.train_result[k])
                val_mean, val_std = np.mean(self.val_result[k]), np.std(self.val_result[k])
                test_mean, test_std = np.mean(self.test_result[k]), np.std(self.test_result[k])
                f.write(f'Train Acc: {train_mean*100:.2f}±{train_std*100:.2f}, Val Acc: {val_mean*100:.2f}±{val_std*100:.2f}, Test Acc: {test_mean*100:.2f}±{test_std*100:.2f}')
                f.write(f'\n')
                if self.args.attack_type == 'poison' and self.args.ptb_rate == 0.25:
                    f.write(f'='*40)
                    f.write(f'\n')
                else:
                    f.write(f'-'*2)
                    f.write(f'\n')

                