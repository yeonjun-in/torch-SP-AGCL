for embedder in SPAGCL_node
do
for dataset in pubmed cs
do
for ptb_rate in 0.0 0.05 0.1 0.15 0.2 0.25
do
python clustering.py --embedder $embedder --dataset $dataset --task clustering --attack meta --ptb_rate $ptb_rate
done
done
done
