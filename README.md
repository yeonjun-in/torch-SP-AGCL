# Similarity-Preserving Adversarial Graph Contrastive Learning


<img src="figs/overall_architecure.jpg" width="500">



### Requirements
* Python version: 3.7.11
* Pytorch version: 1.10.2
* torch-geometric version: 2.0.3
* deeprobust

### How to Run
* To run node classification (reproduce Table 1 in paper, Table 2 and 3 in appendix)

```
sh sh/node.sh
```
* To run link prediction (reproduce Figure 3(b) in paper)

```
sh sh/link.sh
```
* To run node clustering (reproduce Figure 3(c) in paper)
    * You should run node classification before node clustering since we use the embeddings learned in node classification.

```
sh sh/save_emb.sh # save node embedding from the best model of node classification
sh sh/clustering.sh 
```
* To run node classification on heterophilious network (reproduce Table 2 in paper)

```
sh sh/hetero_node.sh
```

