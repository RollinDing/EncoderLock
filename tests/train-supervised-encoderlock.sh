#/bin/bash

# Source model configuration
arch='resnet18'
dataset="mnist"
# Target model configuration
std_dataset="usps"

# Training configuration
epochs=30
print_freq=300
batch_size=64
learning_rate=0.01

# Path configuration
data_path='./data/'
pretrained_path=./pretrained_models/${dataset}_${arch}_30

# Number of critical weights each epoch
N=100
# hyperparameters alpha
alpha=1e3
# Number of epoch for optimization
E=20
# Number of round
R=5
# Data volume
volume=0.1

python src/supervised-encoderlock.py --arch ${arch} --dataset ${dataset} --std-dataset ${std_dataset}\
    --data_path ${data_path} --print_freq ${print_freq}\
     --epochs ${epochs} --learning_rate ${learning_rate} --batch_size ${batch_size}\
    --schedule 15 25  --gammas 0.1 0.1 \
    --resume ${pretrained_path}/checkpoint.pth \
    --N ${N} --E ${E} --R ${R} --alpha ${alpha} --volume ${volume}