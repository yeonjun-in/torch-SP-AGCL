########## general ############
device=1    
seed_n=10
epochs=1000
embedder=SPAGCL_node

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
l1=5.0
l2=3.0
python main.py --embedder $embedder --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node --lambda_1 $l1 --lambda_2 $l2

l1=5.0
l2=3.0
p=0.05
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node --lambda_1 $l1 --lambda_2 $l2

l1=4.0
l2=4.0
p=0.1
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node --lambda_1 $l1 --lambda_2 $l2

l1=4.0
l2=2.0
p=0.15
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node --lambda_1 $l1 --lambda_2 $l2

l1=0.1
l2=5.0
p=0.2
add_edge_rate=0.5
drop_feat_rate=0.7
d_1=0.1
d_2=0.1
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node --lambda_1 $l1 --lambda_2 $l2


l1=0.5
l2=5.0
p=0.25
add_edge_rate=0.5
drop_feat_rate=0.3
d_1=0.1 
d_2=0.2 
lr=0.01 
tau=0.4
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node --lambda_1 $l1 --lambda_2 $l2


dataset=cora
attack=nettack
n_subgraph=3000
lr=0.01
wd=0.0001
d_1=0.1
d_2=0.3
d_3=0.0
add_edge_rate=0.7
drop_feat_rate=0.7
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node
for p in 1 2 3 4 5
do
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node
done

attack=random
n_subgraph=3000
lr=0.01
wd=0.01
d_1=0.2
d_2=0.1
d_3=0.0
add_edge_rate=0.1
drop_feat_rate=0.5
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node
for p in 0.2 0.4 0.6 0.8 1.0
do
python main.py --embedder $embedder --dataset $dataset --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --task node
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
l1=4.0
l2=3.0
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2


l1=2.0
l2=5.0
p=0.05
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

l1=2.0
l2=2.0
p=0.1
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2


l1=2.0
l2=2.0
p=0.15
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2


l1=4.0
l2=5.0
p=0.2
add_edge_rate=0.7 
drop_feat_rate=0.9
d_1=0.2
d_2=0.1
lr=0.01
tau=0.6
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2


l1=2.0
l2=5.0
p=0.25
add_edge_rate=0.9
drop_feat_rate=0.7
d_1=0.2
d_2=0.1
lr=0.01
tau=0.8
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2
 

attack=nettack
n_subgraph=3000
lr=0.001
wd=0.01
d_1=0.1
d_2=0.1
d_3=0.0
add_edge_rate=0.7
drop_feat_rate=0.9
knn=5
tau=0.6
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau 
for p in 1 2 3 4 5
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done

attack=random
n_subgraph=3000
lr=0.001
wd=0.00001
d_1=0.1
d_2=0.4
d_3=0.0
add_edge_rate=0.5
drop_feat_rate=0.7
knn=5
tau=0.8
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.2 0.4 0.6 0.8 1.0
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done


########## pubmed ##############
dataset=pubmed
attack=meta
n_subgraph=1000
lr=0.001
wd=0.00001
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.5
drop_feat_rate=0.7
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done

attack=nettack
n_subgraph=1000
lr=0.005
wd=0.01
d_1=0.5
d_2=0.5
d_3=0.0
add_edge_rate=0.5
drop_feat_rate=0.0
knn=10
tau=0.8
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau 
for p in 1 2 3 4 5
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done

attack=random
n_subgraph=1000
lr=0.001
wd=0.00001
d_1=0.4
d_2=0.1
d_3=0.0
add_edge_rate=0.3
drop_feat_rate=0.0
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.2 0.4 0.6 0.8 1.0
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done


## photo
attack=meta
dataset=photo
n_subgraph=5000
lr=0.01
wd=0.00001
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.1
drop_feat_rate=0.0
knn=10
tau=0.4
seed_n=10
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done


## computers
attack=meta
dataset=computers
n_subgraph=5000
lr=0.01
wd=0.01
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.1
drop_feat_rate=0.0
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done

## cs
attack=meta
dataset=cs
n_subgraph=5000
lr=0.01
wd=0.001
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.7
drop_feat_rate=0.0
knn=10
tau=0.4

l1=0.5
l2=5.0
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

p=0.05
l1=0.5
l2=2.0
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

p=0.1
l1=0.1
l2=5.0
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

p=0.15
l1=0.5
l2=5.0
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

p=0.2
l1=0.5
l2=5.0
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

p=0.25
l1=0.5
l2=5.0
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau --lambda_1 $l1 --lambda_2 $l2

## physics
attack=meta
dataset=physics
n_subgraph=5000
lr=0.01
wd=0.01
d_1=0.3
d_2=0.2
d_3=0.0
add_edge_rate=0.1
drop_feat_rate=0.0
knn=10
tau=0.4
python main.py --embedder $embedder --dataset $dataset --task node --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task node --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs $epochs --add_edge_rate $add_edge_rate --drop_feat_rate $drop_feat_rate --sub_size $n_subgraph --d_1 $d_1 --d_2 $d_2 --d_3 $d_3 --lr $lr --wd $wd --knn $knn --tau $tau
done
