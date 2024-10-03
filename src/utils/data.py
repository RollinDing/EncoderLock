import os
import torch
import torchvision.datasets as dset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from utils.mnistm import MNISTM
from utils.syn import SyntheticDigits
from torch.utils.data import Dataset, Subset
import numpy as np

IMAGE_SIZE = 64
class FAKEMNIST(Dataset):
    def __init__(self, dataset, transform=None):
        self.labels = list(dataset.keys())
        self.samples = []
        self.sample_labels = []
        
        for label, samples in dataset.items():
            self.samples.extend(samples)
            self.sample_labels.extend([label] * len(samples))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.sample_labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    
def prepare_source_data(args):
    dataset = args.dataset
    return prepare_data(args, dataset)

def prepare_target_data(args):
    dataset = args.std_dataset
    return prepare_data(args, dataset)

def prepare_data(args, dataset):
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if dataset == 'cifar10' or dataset == 'stl10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'GTSRB':
        mean = [0.3337, 0.3064, 0.3171]
        std  = [0.2672, 0.2564, 0.2629]
    elif dataset == 'svhn' or dataset == 'syn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'mnist' or dataset == 'mnistm' or dataset == 'emnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'fakemnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'usps':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknown dataset : {}".format(dataset)

    if dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    elif dataset == 'GTSRB':
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset == 'emnist':
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset == 'mnist' or dataset == 'usps' or dataset == 'mnistm' or dataset == 'fakemnist':
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # normalize to match the CIFAR10 dataset
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # normalize to match the CIFAR10 dataset
        ])
    elif dataset == 'stl10' or dataset == 'cifar10' or dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    if dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif dataset == 'mnistm':
        train_data = MNISTM(args.data_path,
                            train=True,
                            transform=train_transform,
                            download=True)
        test_data = MNISTM(args.data_path,
                           train=False,
                           transform=test_transform,
                           download=True)
        num_classes = 10
    elif dataset == 'emnist':
        train_data = dset.EMNIST(args.data_path,
                                split='balanced',
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.EMNIST(args.data_path,
                                split='balanced',
                                train=False,
                                transform=test_transform,
                                download=True)
        num_classes = 47
    elif dataset == 'syn':
        train_data = SyntheticDigits(args.data_path,
                            train=True,
                            transform=train_transform,
                            download=True)
        test_data = SyntheticDigits(args.data_path,
                           train=False,
                           transform=test_transform,
                           download=True)
        num_classes = 10
    elif dataset == 'fakemnist':
        import pickle
        from sklearn.model_selection import train_test_split
        with open('data/fake_mnist.pkl', 'rb') as f:
            dataset = pickle.load(f)
        fakemnist = FAKEMNIST(dataset, transform=train_transform)
        train_data, test_data = train_test_split(fakemnist, test_size=0.2, random_state=42)
        num_classes = 10
    elif dataset == 'usps':
        train_data = dset.USPS(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.USPS(args.data_path,
                                train=False,
                                transform=test_transform,
                                download=True)
        num_classes = 10
    
    elif dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    elif dataset == 'GTSRB':
        train_data = dset.GTSRB(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.GTSRB(args.data_path,
                                 split='test',
                                 transform=test_transform,
                                 download=True)
        num_classes = 43
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    
    return train_loader, test_loader, num_classes

def get_military_dataset(args):
    num_classes = 10
    # Define transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create train and test dataset and dataloader
    train_path = os.path.join(args.data_path, "military", "train")
    train_dataset = ImageFolder(root=train_path, transform=transform)

    test_path = os.path.join(args.data_path, "military", "test")
    test_dataset = ImageFolder(root=test_path, transform=transform)

    val_path = os.path.join(args.data_path, "military", "validation")
    val_dataset = ImageFolder(root=val_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False)

    return train_loader, test_loader, val_loader, num_classes

def get_cifar_dataloader(args, ratio=1.0):
    """
    Get the CIFAR10 dataloader
    """
    num_classes = 10
    # Data loading code for cifar10 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    train_dataset = dset.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(len(train_dataset) * ratio)
    print(f"Using the sample size of {subset_size}.")

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = dset.CIFAR10(
        root=args.data_path,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes

def get_usps_dataloader(args, ratio=1.0):
    """
    Get the USPS dataloader
    """
    num_classes = 10

    # Data loading code for USPS 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    train_dataset = dset.USPS(
        root=args.data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(7291 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = dset.USPS(
        root=args.data_path,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes

def get_mnist_dataloader(args, ratio=1.0):
    """
    Get the MNIST dataloader
    """
    num_classes = 10
    # Data loading code for MNIST 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    train_dataset = dset.MNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(60000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = dset.MNIST(
        root=args.data_path,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes

def get_svhn_dataloader(args, ratio=1.0):
    """
    Get the SVHN dataloader
    """
    num_classes = 10

    # Data loading code for SVHN 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970],
        ),
    ])

    train_dataset = dset.SVHN(
        root=args.data_path,
        split='train',
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(73257 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = dset.SVHN(
        root=args.data_path,
        split='test',
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes

def get_mnistm_dataloader(args, ratio=1.0):
    """
    Get the MNISTM dataloader
    """
    num_classes = 10
    # Data loading code for MNISTM 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.1307],
            std=[0.3081],
        ),
    ])

    train_dataset = MNISTM(
        root=args.data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(60000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = MNISTM(
        root=args.data_path,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes

def get_syn_dataloader(args, ratio=1.0):
    num_classes = 10
    """
    Get the Synthetic Digits dataloader
    """
    # Data loading code for Synthetic Digits 
    train_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.transforms.ToTensor(),
    ])

    train_dataset = SyntheticDigits(
        root=args.data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Define the size of the subset
    subset_size = int(60000 * ratio)# for example, 5000 samples

    # Create a random subset for training
    indices = np.random.permutation(60000)
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    val_dataset = SyntheticDigits(
        root=args.data_path,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    return train_loader, val_loader, num_classes

def get_imagenet_dataloader(args, ratio=1.0):
    """
    Get the ImageNet dataloader
    """
    num_classes = 1000
    # Data loading code for ImageNet 
    train_transform = transforms.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.transforms.ToTensor(),
        transforms.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_dataset = dset.ImageFolder(root=args.data_path+'imagenet-mini/val',
                                      transform=val_transform)
    train_dataset = dset.ImageFolder(root=args.data_path+'imagenet-mini/train',
                                        transform=train_transform)

    train_dataset = dset.ImageNet(root=args.data_path+'imagenet', split='train',
                                    transform=train_transform)
    val_dataset = dset.ImageNet(root=args.data_path+'imagenet', split='val',
                                    transform=val_transform)

    subset_size = int(len(train_dataset) * ratio)

    # indices = np.random.permutation(len(train_dataset))
    # train_indices = indices[:subset_size]
    # train_subset = Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes

def get_cars_dataloader(args, ratio=1.0):
    """
    Get the Stanford Cars dataloader
    """
    num_classes = 196
    # Data loading code for Stanford Cars 
    RESCALE_SIZE = 224
    train_transforms = transforms.Compose([transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
                                    transforms.RandomHorizontalFlip(),                   
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    test_transforms = transforms.Compose([transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    train_dataset = dset.ImageFolder(root=args.data_path+'stanford_cars/train',
                                        transform=train_transforms)
    val_dataset = dset.ImageFolder(root=args.data_path+'stanford_cars/test',
                                        transform=test_transforms)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, num_classes
    
def get_imagenette_dataloader(args, ratio=0.1):
    # Define transformations
    num_classes = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transform with augmentation
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create train and test dataset and dataloader
    data_path = args.data_path + "imagenette2/train"
    train_dataset = ImageFolder(root=data_path, transform=transform)

    # Define the size of the subset
    subset_size = int(len(train_dataset)*ratio)
    print(f"Using sample size of {subset_size}.")
                      
    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    data_path = args.data_path + "imagenette2/val"
    test_dataset = ImageFolder(root=data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader, num_classes

def get_imagewoof_dataloader(args, ratio=0.1):
    # Define transformations
    num_classes = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create train and test dataset and dataloader
    data_path = args.data_path + "imagewoof2/train"
    train_dataset = ImageFolder(root=data_path, transform=transform)
    # Define the size of the subset
    subset_size = int(len(train_dataset)*ratio)
    print(f"Using sample size of {subset_size}.")
                      
    # Create a random subset for training
    indices = np.random.permutation(len(train_dataset))
    train_indices = indices[:subset_size]
    train_subset = Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    data_path = args.data_path + "imagewoof2/val"
    test_dataset = ImageFolder(root=data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader, num_classes

def get_tiny_imagenet_dataloader(args, ratio=1.0):
    """
    Get the Tiny ImageNet dataloader
    """
    num_classes = 200
    # Data loading code for Tiny ImageNet 
    image_size = args.image_size
    train_transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.RandomHorizontalFlip(),                   
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])])
    
    test_transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])])
    
    train_path = os.path.join(args.data_path, "tiny-imagenet-200", "train")
    test_path = os.path.join(args.data_path, "tiny-imagenet-200", "test")
    val_path = os.path.join(args.data_path, "tiny-imagenet-200", "val")

    train_dataset = ImageFolder(root=train_path, transform=train_transforms)
    val_dataset = ImageFolder(root=val_path, transform=train_transforms)
    test_dataset = ImageFolder(root=test_path, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, val_loader, test_loader, num_classes
    

from PIL import Image
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # the image file is the JPEG file in the root directory
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Add Gaussian noise to the tensor
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        noisy_tensor = torch.clamp(noisy_tensor, 0, 255)
        
        return noisy_tensor
    
def get_synthetic_military_dataset(args):
    # Define the transformations to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=5.), # Add Gaussian noise to the images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an instance of the custom dataset
    dataset = UnlabeledImageDataset(root_dir=args.data_path+"synthetic_military-3", transform=transform)

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    return dataloader