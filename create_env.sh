conda create -n rgnn python=3.7

conda activate rgnn
conda install pytorch torchvision torchaudio -c pytorch
conda install pyg -c pyg -c conda-forge

pip install deeprobust
pip install tensorboard
pip install seaborn
pip install matplotlib
