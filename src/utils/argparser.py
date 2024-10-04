import argparse

def parse_args():
    ################# Options ##################################################
    ############################################################################
    parser = argparse.ArgumentParser(
        description='Training network for image classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_path',
                        default='/home/ruyi/dnn-transferability/dnn-transferability-fp/data/',
                        type=str,
                        help='Path to dataset')

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'usps', 'mnistm', 'syn', 'fakemnist', 'GTSRB', 'emnist', 'military'],
        help='Choose between Cifar10/100 and ImageNet, and digits dataset.')

    parser.add_argument(
        '--std-dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'usps', 'mnistm', 'syn', 'fakemnist', 'GTSRB', 'emnist', 'military'],
        help='Choose between Cifar10/100 and ImageNet as student downstream dataset.')

    parser.add_argument('--arch',
                        metavar='ARCH',
                        default='lbcnn',
                        help='model architecture;')

    # Optimization options
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to train.')

    parser.add_argument('--optimizer',
                        type=str,
                        default='SGD',
                        choices=['SGD', 'Adam', 'YF'])
    
    parser.add_argument('--image-size',
                        type=int,
                        default=32,
                        help='Image size (default: 32)')

    parser.add_argument('--percent',
                        type=float,
                        default=0.5,
                        help='Percentage data use for transfer learning finetuning')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size.')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='The Learning Rate.')

    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')

    parser.add_argument('--decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay (L2 penalty).')

    parser.add_argument('--schedule',
                        type=int,
                        nargs='+',
                        default=[80, 120],
                        help='Decrease learning rate at these epochs.')

    parser.add_argument(
        '--gammas',
        type=float,
        nargs='+',
        default=[0.1, 0.1],
        help=
        'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
    )

    # Checkpoints
    parser.add_argument('--print_freq',
                        default=100,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 200)')

    parser.add_argument('--save_path',
                        type=str,
                        default='./save/',
                        help='Folder to save checkpoints and log.')

    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument(
        '--fine_tune',
        dest='fine_tune',
        action='store_true',
        help='fine tuning from the pre-trained model, force the start epoch be zero'
    )

    parser.add_argument('--model_only',
                        dest='model_only',
                        action='store_true',
                        help='only save the model without external utils_')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')

    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='device range [0,ngpu-1]')

    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='number of data loading workers (default: 2)')

    # random seed
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

    parser.add_argument('--n_iter',
                        type=int,
                        default=20,
                        help='number of attack iterations')
    
    parser.add_argument('--volume', type=float, default=0.1, help='data volume of transfer learning')

    # optimization arguments
    parser.add_argument('--N', type=int, default=100, help='number of weights selected')
    parser.add_argument('--M', type=int, default=1, help='number of important layers selected')
    parser.add_argument('--optim-lr', type=float, default=0.01, help='learning rate for optimization')
    parser.add_argument('--U', type=float, default=10, help='upper bound of target loss')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for the regularization term')
    parser.add_argument('--E', type=int, default=10, help='number of epochs for optimization')
    parser.add_argument('--R', type=int, default=10, help='number of round per epoch')
    parser.add_argument('--level', type=str, default='example-supervised', 
                        choices=['example-supervised', 'example-unsupervised', 'supervised', 'unsupervised'], help='level of evaluation')

    return parser.parse_args()