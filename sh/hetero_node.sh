########## general ############
device=0
seed_n=10
epochs=1000
embedder=SPAGCL_node

## chameleon
attack=random
dataset=chameleon
n_subgraph=3000
lr=0.01
wd=0.00001
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.3
drop_feat_rate=0.1
knn=10
tau=0.4
python main.py --embedder $embedder --task node --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau


## squirrel
attack=random
dataset=squirrel
n_subgraph=3000
lr=0.01
wd=0.00001
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.1
drop_feat_rate=0.3
knn=10
tau=0.4
python main.py --embedder $embedder --task node --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau


# Actor
attack=random
dataset=actor
n_subgraph=3000
lr=0.01
wd=0.00001
d_1=0.1
d_2=0.1
d_3=0.0
add_edge_rate=0.3
drop_feat_rate=0.3
knn=10
tau=0.4
python main.py --embedder $embedder --task node --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau

## texas
attack=random
dataset=texas
n_subgraph=3000
lr=0.05
wd=0.00001
d_1=0.5
d_2=0.5
d_3=0.0
add_edge_rate=0.7
drop_feat_rate=0.9
knn=10
tau=0.4
python main.py --embedder $embedder --task node --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau


## cornell
attack=random
dataset=cornell
n_subgraph=3000
lr=0.05
wd=0.00001
d_1=0.4
d_2=0.3
d_3=0.0
add_edge_rate=0.7
drop_feat_rate=0.5
knn=10
tau=0.4
python main.py --embedder $embedder --task node --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau

## wisconsin
attack=random
dataset=wisconsin
n_subgraph=3000
lr=0.05
wd=0.00001
d_1=0.2
d_2=0.4
d_3=0.0
add_edge_rate=0.7
drop_feat_rate=0.0
knn=10
tau=0.2
python main.py --embedder $embedder --task node --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
