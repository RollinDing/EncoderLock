import torch
import torch.nn as nn
import torchvision.models as models
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(ResNetFeatureExtractor, self).__init__()
        # Here we get all layers except the last fully connected layer.
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])

    def forward(self, x):
        return self.features(x)
