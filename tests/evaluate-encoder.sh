#/bin/bash

# Choose between arch [vgg11, resnet18]
arch='resnet18'
# Source model configuration
dataset="mnist"
# Target model configuration
std_dataset="usps"
# Choose between ['example-supervised', 'example-unsupervised', 'supervised', 'unsupervised']
level='example-supervised'


# Training configuration
epochs=30
print_freq=300
batch_size=128
learning_rate=0.01
volume=0.1

# Path configuration
data_path='./data/'
pretrained_path=./pretrained_models/${dataset}_${arch}_${epochs}

python src/evaluate/evaluate-encoder.py --arch ${arch} --dataset ${dataset} --std-dataset ${std_dataset}\
    --data_path ${data_path} --print_freq ${print_freq}\
    --epochs ${epochs} --learning_rate ${learning_rate} --batch_size ${batch_size}\
    --schedule 15 25  --gammas 0.1 0.1 \
    --resume ${pretrained_path}/checkpoint.pth --level ${level} --volume ${volume}