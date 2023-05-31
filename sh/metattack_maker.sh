# photo
dataset=photo
sub_size=3000
for ((i=1; i<=100; i++))
do
python ./attack_maker/generate_metattack.py --dataset $dataset --device 6 --sub_size 3000 --ptb_n 300 --seed $i 
done

# computers
dataset=computers
sub_size=3000
for ((i=91; i<=200; i++))
do
python ./attack_maker/generate_metattack.py --dataset $dataset --device 7 --sub_size 3000 --ptb_n 300 --seed $i 
done

# cs
dataset=cs
sub_size=3000
for ((i=1; i<=100; i++))
do
python ./attack_maker/generate_metattack.py --dataset $dataset --device 7 --sub_size 3000 --ptb_n 200 --seed $i
done

# physics
dataset=physics
sub_size=3000
for ((i=81; i<=300; i++))
do
python ./attack_maker/generate_metattack.py --dataset $dataset --device 6 --sub_size 3000 --ptb_n 200 --seed $i 
done
