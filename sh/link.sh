########## general ############
device=1
seed_n=3
epochs=1000
embedder=SPAGCL_link
verbose=100

########## cora ##############
dataset=cora
attack=meta
n_subgraph=3000
lr=0.005
wd=0.01
d_1=0.2
d_2=0.3
d_3=0.0
add_edge_rate=0.5
drop_feat_rate=0.7
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --task link --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task link --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done

########## citeseer ##############
dataset=citeseer
attack=meta
n_subgraph=3000
lr=0.01
wd=0.00001
d_1=0.2
d_2=0.1
d_3=0.0
add_edge_rate=0.1
drop_feat_rate=0.9
knn=10
tau=0.6
python main.py --embedder $embedder --dataset $dataset --task link --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task link --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done
