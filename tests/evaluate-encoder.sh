#/bin/bash

# Source model configuration
arch="vgg11"
dataset="mnist"

# Target model configuration
std_dataset="usps"

# Training configuration
epochs=30
print_freq=300
batch_size=128
learning_rate=0.01

# Path configuration
data_path='./data/'
pretrained_path=./pretrained_models/${dataset}_${arch}_${epochs}

# Ns=(1 2 3 5 8)
M=1
U=10
optim_lr=1e-2
# alphas=(10 100 1000 1e4 1e5)
alpha=10
E=200
R=5
volume=0.1
N=100

python src/evaluate/evaluate-encoder.py --arch ${arch} --dataset ${dataset} --std-dataset ${std_dataset}\
    --data_path ${data_path} --print_freq ${print_freq}\
    --epochs ${epochs} --learning_rate ${learning_rate} --batch_size ${batch_size}\
    --schedule 15 25  --gammas 0.1 0.1 \
    --resume ${pretrained_path}/checkpoint.pth \
    --N ${N} --M ${M} --U ${U} --optim-lr ${optim_lr} --E ${E} --R ${R} --alpha ${alpha}