#/bin/bash

# Choose between arch [vgg11, resnet18]
arch='vgg11'
# Source model configuration
dataset="mnist"
# Target model configuration
std_dataset="svhn"
# Choose between ['example-supervised', 'example-unsupervised', 'supervised', 'unsupervised']
level='example-unsupervised' 


# Training configuration
epochs=30
print_freq=300
batch_size=128
learning_rate=0.01

# Path configuration
data_path='./data/'
pretrained_path=./pretrained_models/${dataset}_${arch}_${epochs}

M=1
U=10
optim_lr=1e-2
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
    --N ${N} --M ${M} --U ${U} --optim-lr ${optim_lr} --E ${E} --R ${R} --alpha ${alpha} --level ${level}