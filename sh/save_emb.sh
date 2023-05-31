seed_n=10
attack=meta
device=0
for dataset in cora citeseer pubmed photo computers cs physics
do
for embedder in SPAGCL_node
do
python main.py --embedder $embedder --dataset $dataset --task node --save_embed --attack_type evasive --attack $attack --device $device --seed_n $seed_n --epochs 0
for p in 0.05 0.1 0.15 0.2 0.25
do
python main.py --embedder $embedder --dataset $dataset --task node --save_embed --ptb_rate $p --attack_type poison --attack $attack --device $device --seed_n $seed_n --epochs 0
done
done
done