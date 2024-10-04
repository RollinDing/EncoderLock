"""
This code snippet is used to evaluate the transferability of feature extractors.
Including:
    1. The feature extractor performance on source domain;
    2. The feature extractor performance on target domain;
    3. The feature extractor performance on other domains.
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Conv2d, Linear

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.argparser import parse_args
from utils.data import prepare_source_data, prepare_target_data, prepare_data
from utils.utils import *
import logging, time, os, copy

# Load the arguments
args = parse_args()
LEVEL = args.level

def create_logging_files(args):
    """
    Create the logging files
    """
    # Create the logging directory
    log_dir = f"logs/{LEVEL}/" + f'evaulation-{args.arch}--{args.dataset}--{args.std_dataset}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the logging file based on time stamp
    logging.basicConfig(
        filename=os.path.join(log_dir, f'evaluation-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
class Identity(nn.Module):
    def forward(self, x):
        return x
    
def remove_bn_from_model(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, Identity())
        else:
            remove_bn_from_model(child)

def build_model(args):
    model_name = args.arch
    if model_name == 'resnet18':
        model = models.resnet18()
        remove_bn_from_model(model)
        model.fc = nn.Linear(512, 10)
    elif model_name == 'vgg11':
        model = models.vgg11()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
    return model

def build_feature_extractor(args):
    model_name = args.arch
    if model_name == 'resnet18':
        model = build_model(args)
        feature_extractor = ResNetFeatureExtractor(model)
    elif model_name == 'vgg11':
        model = build_model(args)
        feature_extractor = model.features
    return feature_extractor


def load_feature_extractor(args):
    """
    Load the feature extractor
    """
    feature_extractor = build_feature_extractor(args)
    # You can select supervised or unsupervised here
    feature_extractor_path = "modified_models/" + f'{LEVEL}/' + f'{args.arch}--{args.dataset}--{args.std_dataset}' + f'/feature_extractor.pth'
    print(feature_extractor_path)
    if not os.path.exists(feature_extractor_path):
        exit('The feature extractor does not exist!')
    else:
        feature_extractor.load_state_dict(torch.load(feature_extractor_path))
    return feature_extractor

def check_weight_update(prev_model, model):
    # 1. Store initial weights
    initial_weights = {}
    for name, param in prev_model.named_parameters():
        initial_weights[name] = param.clone()  # .clone() is important to get a copy, not a reference

    # 2. Compare with new weights
    changed_weights = 0
    total_weights = 0
    for name, param in model.named_parameters():
        total_weights += torch.numel(param)
        changed_weights += torch.sum(initial_weights[name] != param).item()
    
    logging.info(f"Total weights: {total_weights}, Changed weights: {changed_weights}, Percentage: {changed_weights / total_weights}")

def check_weight_changed(args, feature_extractor):
    model_name = args.arch
    org_model = build_model(args)
    # load the original model 
    org_model.load_state_dict(torch.load(args.resume))
    if model_name == 'vgg11':
        check_weight_update(org_model.features, feature_extractor)
    elif model_name == 'resnet18':
        check_weight_update(ResNetFeatureExtractor(org_model), feature_extractor)

def get_downstream_classifier(args, num_classes):
    if args.arch == 'vgg11':
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif args.arch == 'resnet18':
        classifier = nn.Linear(512, num_classes)
    return classifier

def transfer_learning(args, feature_extractor, classifier, train_loader, test_loader, criterion, device, model_name, verbose=True, patience=10):
    """
    Transfer model from source to target, training the target downstream tasks
    """
    data_volume = args.volume
    num_batches = len(train_loader)
    num_train_batches = int( data_volume * num_batches)
    num_epochs = 100

    feature_extractor.train()
    classifier.train()

    for param in feature_extractor.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = True

    learning_rate=1e-4
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 75], gamma=0.1)
    # Early stopping setup
    best_acc = 0.0
    counter = 0  # Counts epochs without improvement
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        avg_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= num_train_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            features = feature_extractor(inputs)
            if model_name == 'vgg11':
                features = nn.AdaptiveAvgPool2d((7, 7))(features)
                features = features.view(features.size(0), -1)
            elif model_name == 'resnet18':
                features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.view(features.size(0), -1)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss = criterion(outputs, labels)
            avg_loss += loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        avg_loss /= num_train_batches
        
        # Evaluate on validation/test set to check for early stopping
        # Replace this with your evaluation logic if different
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        best_loss = 1000000
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = feature_extractor(inputs)
                if model_name == 'vgg11':
                    features = nn.AdaptiveAvgPool2d((7, 7))(features)
                    features = features.view(features.size(0), -1)
                elif model_name == 'resnet18':
                    features = nn.AdaptiveAvgPool2d((1, 1))(features)
                    features = features.view(features.size(0), -1)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_loss += criterion(outputs, labels).item()
        
        scheduler.step()
        val_acc = 100 * val_correct / val_total
        val_loss /= len(test_loader)

        # Check for early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0  # Reset the counter
            # Optionally save the model here
            best_classifier = copy.deepcopy(classifier)
        else:
            counter += 1
            
        if verbose:
            print(f'==> Epoch: {epoch} | Loss: {loss.item()} | Train Accuracy: {100 * correct / total:.2f}% | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% ')
            
        if counter >= patience:
            print("Early stopping triggered.")            
            break  # Stop the training

    return best_classifier

def do_evaluate_feature_extractor(feature_extractor, classifier, test_loader, criterion, device, model_name):
    feature_extractor.eval()
    classifier.eval()  
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            features = feature_extractor(inputs)
            if model_name == 'vgg11':
                features = nn.AdaptiveAvgPool2d((7, 7))(features)
                features = features.view(features.size(0), -1)
            elif model_name == 'resnet18':
                features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.view(features.size(0), -1)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    logging.info('==> Test Loss: {:.4f} | Accuracy: {:.2f}%\n'.format(avg_loss, accuracy))

    return avg_loss, accuracy

def evaluate_feature_extractor(args, feature_extractor):
    """
    Evaluate the feature extractor on different domains
    """
    model_name = args.arch
    # Prepare the data
    source_train_loader, source_test_loader, source_num_classes = prepare_source_data(args)
    target_train_loader, target_test_loader, target_num_classes = prepare_target_data(args)

    # Get the downstream classifier
    source_downstream_classifier = get_downstream_classifier(args, source_num_classes)
    target_downstream_classifier = get_downstream_classifier(args, target_num_classes)

    # Assign downstream classifier to device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    source_downstream_classifier = source_downstream_classifier.to(device)
    target_downstream_classifier = target_downstream_classifier.to(device)

    # transfer learning the downstream classifier
    criterion = nn.CrossEntropyLoss()
    source_downstream_classifier = transfer_learning(args, feature_extractor, source_downstream_classifier, source_train_loader, source_test_loader, criterion, device, model_name)
    target_downstream_classifier = transfer_learning(args, feature_extractor, target_downstream_classifier, target_train_loader, target_test_loader, criterion, device, model_name)

    # Evaluate the feature extractor on source domain
    logging.info("Evaluating on the source domain")
    do_evaluate_feature_extractor(feature_extractor, source_downstream_classifier, source_test_loader, criterion, device, model_name)
    logging.info("Evaluating on the target domain")
    do_evaluate_feature_extractor(feature_extractor, target_downstream_classifier, target_test_loader, criterion, device, model_name)


if __name__ == '__main__':
    # Create the logging files
    create_logging_files(args)
    logging.info('Start to evaluate the feature extractor.')
    logging.info('The arguments are: {}'.format(args))

    # Load the feature extractor
    feature_extractor = load_feature_extractor(args)

    # Number of changed weights 
    check_weight_changed(args, feature_extractor)

    # Evaluate the feature extractor
    evaluate_feature_extractor(args, feature_extractor)
