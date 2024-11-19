"""
This the file to build encoderlock without knowing the target dataset (especially the label of the target domain)

The code design is as follows:
    1. Load the pretrained general encoder 
    2. Define the source and target domain.
    3. Sample from the source domain's datasets and sample from the target domain samples (without labels)
    4. build the loss function and search for critical weights.
    5. Update the critical weights and evaluate the transferability
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Conv2d, Linear

from utils.argparser import parse_args
from utils.data import prepare_source_data, prepare_target_data
from utils.utils import *
import logging, time, os, copy
from torchvision import transforms

IMAGE_SIZE=32
# Define the data augmentation transforms
augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def augment_batch_tensor(batch_data):
    # Assuming you have a batch of image tensors called 'batch_data'
    batch_size, channels, height, width = batch_data.size()

    # Apply data augmentation transforms to the batch
    augmented_batch = torch.stack([augmentation(img) for img in batch_data])

    # Reshape the augmented batch back to (batch_size, channels, height, width)
    augmented_batch = augmented_batch.view(batch_size, channels, height, width)
    return augmented_batch

class Identity(nn.Module):
    def forward(self, x):
        return x
    
def remove_bn_from_model(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, Identity())
        else:
            remove_bn_from_model(child)

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_feature_extractor(args, model):
    save_path = "modified_models/unsupervised/" + f'{args.arch}--{args.dataset}--{args.std_dataset}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + f'/feature_extractor.pth')

def load_model(args, model):
    model.load_state_dict(torch.load(args.resume))
    return model

def check_weight_update(prev_model, model):
    # 1. Store initial weights
    initial_weights = {}
    for name, param in prev_model.named_parameters():
        initial_weights[name] = param.clone()  # .clone() is important to get a copy, not a reference

    # You can also manually run a few optimization steps here if you want

    # 3. Compare with new weights
    changed_weights = 0
    total_weights = 0
    for name, param in model.named_parameters():
        total_weights += torch.numel(param)
        changed_weights += torch.sum(initial_weights[name] != param).item()
    
    print(f"Total weights: {total_weights}, Changed weights: {changed_weights}")
    return changed_weights

def evaluate_feature_extractor(feature_extractor, classifier, test_loader, criterion, device, model_name='vgg11'):
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
            elif model_name == 'mobilenetv2':
                features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.view(features.size(0), -1)
            elif model_name == 'densenet121':
                features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.view(features.size(0), -1)
            elif model_name == 'alexnet':
                features = nn.AdaptiveAvgPool2d((6, 6))(features)
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

def transfer_learning(args, feature_extractor, classifier, train_loader, test_loader, criterion, device, model_name='vgg11', learning_rate=1e-4, verbose=True, patience=3, data_volume=0.1):
    """
    Transfer model from source to target, training the target downstream tasks
    """
    num_batches = len(train_loader)
    num_train_batches = int(data_volume * num_batches)
    num_epochs = args.epochs

    feature_extractor.train()
    classifier.train()

    for param in feature_extractor.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 75], gamma=0.1)
    
    # Early stopping setup
    best_acc = 0.0
    counter = 0  # Counts epochs without improvement
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
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
            elif model_name == 'mobilenetv2':
                features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.view(features.size(0), -1)
            elif model_name == 'densenet121':
                features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.view(features.size(0), -1)
            elif model_name == 'alexnet':
                features = nn.AdaptiveAvgPool2d((6, 6))(features)
                features = features.view(features.size(0), -1)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate on validation/test set to check for early stopping
        # Replace this with your evaluation logic if different
        val_correct = 0
        val_total = 0
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
                elif model_name == 'mobilenetv2':
                    features = nn.AdaptiveAvgPool2d((1, 1))(features)
                    features = features.view(features.size(0), -1)
                elif model_name == 'densenet121':
                    features = nn.AdaptiveAvgPool2d((1, 1))(features)
                    features = features.view(features.size(0), -1)
                elif model_name == 'alexnet':
                    features = nn.AdaptiveAvgPool2d((6, 6))(features)
                    features = features.view(features.size(0), -1)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Check for early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0  # Reset the counter
            # Optionally save the model here
        else:
            counter += 1
            
        if verbose:
            print(f'==> Epoch: {epoch} | Loss: {loss.item()} | Train Accuracy: {100 * correct / total:.2f}% | Val Accuracy: {val_acc:.2f}%')
            
        if counter >= patience:
            print("Early stopping triggered.")
            break  # Stop the training
    
    return classifier

def adjust_learning_rate(args, optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features, augmented_features):
        # Compute pairwise cosine similarities between features
        similarities = torch.matmul(features, augmented_features.t()) / self.temperature

        # Create a mask for positive pairs (diagonal elements)
        mask_pos = torch.eye(features.size(0), dtype=torch.bool, device=features.device)

        # Create a mask for negative pairs (off-diagonal elements)
        mask_neg = ~mask_pos

        # Apply masks and compute sums for positive and negative pairs
        exp_similarities = torch.exp(similarities)
        exp_pos = exp_similarities[mask_pos].unsqueeze(1)  # Keep dims for broadcasting
        sum_neg = exp_similarities.masked_select(mask_neg).view(features.size(0), -1).sum(dim=1, keepdim=True)

        # Compute the contrastive loss
        loss = -torch.log(exp_pos / (exp_pos + sum_neg)).mean()

        return loss

def upper_level_optimization(feature_extractor, source_classifier, source_train_loader, target_train_loader, mask, device, model_name):
    """
    Given a pretrained feature extractor and two classifiers, we are going to find the critical weights for the target domain
    The definition of critical weights is that the weights are important for the target domain but not for the source domain, 
    Please note that for the unsupervised encoderlock, we do not have the label for the target domain, so we are going to use the
    contrastive loss of the target domain to find the critical weights.
    """

    # The criterion is important for the optimization
    criterion = nn.CrossEntropyLoss()

    # Set the feature extractor and classifiers to evaluation mode
    feature_extractor.eval()
    source_classifier.eval()

    # Get the weights gradient for the feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = True

    # Get the data for optimization 
    num_batches = 10

    source_data_list = []
    source_labels_list = []
    target_data_list = []

    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    for _ in range(num_batches):
        try:
            source_data, source_labels = next(source_iter)
            source_data_list.append(source_data)
            source_labels_list.append(source_labels)
            
            # only use the data if target iter has label
            # target_data, target_label = next(target_iter)
            target_data, _ = next(target_iter)
            # select the first element of the tuple
            target_data_list.append(target_data)
        except StopIteration:
            # Handle case where there may not be enough batches in the dataloader
            break

    # Concatenate all batches
    source_data = torch.cat(source_data_list).to(device)
    source_labels = torch.cat(source_labels_list).to(device)
    target_data = torch.cat(target_data_list).to(device)

    for m in feature_extractor.modules():
        if isinstance(m, Conv2d) or isinstance(m, Linear):
            if m.weight.grad is not None:
                m.weight.grad.data.zero_()

    source_data, source_labels = source_data.to(device), source_labels.to(device)
    target_data = target_data.to(device)

    # Extract features
    source_features = feature_extractor(source_data)
    if model_name == 'vgg11':
        source_features = nn.AdaptiveAvgPool2d((7, 7))(source_features)
        source_features = source_features.view(source_features.size(0), -1)
    elif model_name == 'resnet18':
        source_features = nn.AdaptiveAvgPool2d((1, 1))(source_features)
        source_features = source_features.view(source_features.size(0), -1)
    source_outputs = source_classifier(source_features)
    
    _, source_predicted = source_outputs.max(1)
    source_correct = source_predicted.eq(source_labels).sum().item()
    source_total = source_labels.size(0)
    print("Source accuracy: ", source_correct / source_total)

    # Compute loss for the source
    loss_source = criterion(source_outputs, source_labels)
    loss_source.backward(retain_graph=True)

    # Store source gradients
    source_gradients = {}
    for name, param in feature_extractor.named_parameters():
        if 'bias' not in name:
            if param.grad is not None:
                source_gradients[name] = param.grad.clone()
        param.grad.zero_()  # Reset gradients

    # Compute loss for the target
    target_features = feature_extractor(target_data)
    if model_name == 'vgg11':
        target_features = nn.AdaptiveAvgPool2d((7, 7))(target_features)
        target_features = target_features.view(target_features.size(0), -1)
    elif model_name == 'resnet18':
        target_features = nn.AdaptiveAvgPool2d((1, 1))(target_features)
        target_features = target_features.view(target_features.size(0), -1)
   
    augmented_target_data = augment_batch_tensor(target_data)
    augmented_target_data = augmented_target_data.to(device)
    augmented_target_features = feature_extractor(augmented_target_data)
    if model_name == 'vgg11':
        augmented_target_features = nn.AdaptiveAvgPool2d((7, 7))(augmented_target_features)
        augmented_target_features = augmented_target_features.view(augmented_target_features.size(0), -1)
    elif model_name == 'resnet18':
        augmented_target_features = nn.AdaptiveAvgPool2d((1, 1))(augmented_target_features)
        augmented_target_features = augmented_target_features.view(augmented_target_features.size(0), -1)

    

    # normalize the features
    target_features = F.normalize(target_features, p=2, dim=1)
    augmented_target_features = F.normalize(augmented_target_features, p=2, dim=1)    

    contrastive_loss = ContrastiveLoss(temperature=0.8)
    loss_target = contrastive_loss(target_features, augmented_target_features)
    loss_target.backward()

    # Store target gradients for each layer
    target_gradients = {}
    for name, param in feature_extractor.named_parameters():
        if 'bias' not in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:
            if param.grad is not None:
                target_gradients[name] = param.grad.clone()

    # Now the weighted scores is the difference between the target and source gradients
    # The target weights should be the one with largest gradient in target gradients, and the one with smallest gradient in source gradients
    # We also require the important weights has relatively large magnitude to ensure that the weights are important to the model.
    weight_scores = {}
    weight_threshold = 1e-4
    for name, param in feature_extractor.named_parameters():
        if name in source_gradients and name in target_gradients:
            selected_weights = torch.where(param.abs()>weight_threshold)
            weight_scores[name] = ((target_gradients[name].abs() / (source_gradients[name].abs() + 1e-8)))[selected_weights] # Weight score (gradient ratio) vs gradient times magnitude.

    # Select the weights with the highest impact (this is a simple example for selecting top N weights)
    N = args.N  # Number of weights to select

    # For selecting top M layers based on their gradient impact
    M = args.M  # adjust based on your preference

    layer_scores = {}
    for name, score in weight_scores.items():
        layer_scores[name] = score.max().item()

    # Sort layers by their scores
    sorted_layer_scores = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)

    # Get top M layers
    top_m_layers = [layer[0] for layer in sorted_layer_scores[:M]]
    logging.info(f"Selected layers: {top_m_layers}")

    # for each layer select N weights
    all_scores = {}

    for layer in top_m_layers:
        scores = weight_scores[layer].flatten().tolist()
        sorted_scores_with_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        # Get top N indices
        top_n_indices = [index for index, _ in sorted_scores_with_indices[:N]]
        all_scores[layer] = top_n_indices
        print(f"Selected weights for layer {layer}: {top_n_indices}")
    
    # Create a mask to represent which weights are critical
    new_mask = {}
    for name, param in feature_extractor.named_parameters():
        if name in top_m_layers:
            mask_flat = torch.zeros_like(param.flatten()).to(device)

            # Set mask for top N scores
            top_n_indices = all_scores[name]
            mask_flat[top_n_indices] = 1.

            # Reshape the flattened mask back to the shape of the original tensor
            new_mask[name] = mask_flat.reshape(param.shape)
        else:
            new_mask[name] = torch.zeros_like(param, device=device)
    
    # Combine the new mask with the old mask
    for name, param in feature_extractor.named_parameters():
        if name in mask:
            new_mask[name] += mask[name]
    
    # binarize new mask
    for name, param in feature_extractor.named_parameters():
        if name in mask:
            new_mask[name][new_mask[name]>0]=1

    return new_mask

def update_critical_weights(feature_extractor, source_classifier, target_classifier, mask, source_loader, target_loader, device, model_name, epochs=5):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    learning_rate = args.optim_lr
    
    source_classifier.eval()

    # Enable requires_grad for all parameters as we will mask gradients directly
    for param in feature_extractor.parameters():
        param.requires_grad = True

    # Define an optimizer. We are only optimizing the feature_extractor's parameters.
    # The feature extractor is still evaluation mode as we do not want to modify the batchnorm.
    optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)

    # Total number of batch we are going to use during update critical weights
    nbatch = 10
    for epoch in range(epochs):
        total_loss = 0.0
        for b, ((source_data, source_labels), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
            # Only update for first 300 batch
            if b >= nbatch:
                break
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            # target_data = target_data[0] if isinstance(target_data, tuple) else target_data
            target_data = target_data.to(device)
            # target_labels = target_labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            prev_model = copy.deepcopy(feature_extractor)
            # Forward Pass
        
            source_features = feature_extractor(source_data)
            if model_name == 'vgg11':
                source_features = nn.AdaptiveAvgPool2d((7, 7))(source_features)
                source_features = source_features.view(source_features.size(0), -1)
            elif model_name == 'resnet18':
                source_features = nn.AdaptiveAvgPool2d((1, 1))(source_features)
                source_features = source_features.view(source_features.size(0), -1)           
            source_outputs = source_classifier(source_features)

            target_features = feature_extractor(target_data)
            if model_name == 'vgg11':
                target_features = nn.AdaptiveAvgPool2d((7, 7))(target_features)
                target_features = target_features.view(target_features.size(0), -1)
            elif model_name == 'resnet18':
                target_features = nn.AdaptiveAvgPool2d((1, 1))(target_features)
                target_features = target_features.view(target_features.size(0), -1)
            target_outputs = target_classifier(target_features)

            batch_size = target_features.size(0)
            # flatten the features
            target_features = target_features.view(batch_size, -1)

            augmented_target_data = augment_batch_tensor(target_data)
            augmented_target_data = augmented_target_data.to(device)
            augmented_target_features = feature_extractor(augmented_target_data)
            if model_name == 'vgg11':
                augmented_target_features = nn.AdaptiveAvgPool2d((7, 7))(augmented_target_features)
                augmented_target_features = augmented_target_features.view(augmented_target_features.size(0), -1)
            elif model_name == 'resnet18':
                augmented_target_features = nn.AdaptiveAvgPool2d((1, 1))(augmented_target_features)
                augmented_target_features = augmented_target_features.view(augmented_target_features.size(0), -1)

            augmented_source_data = augment_batch_tensor(source_data)
            augmented_source_data = augmented_source_data.to(device)
            augmented_source_features = feature_extractor(augmented_source_data)
            if model_name == 'vgg11':
                augmented_source_features = nn.AdaptiveAvgPool2d((7, 7))(augmented_source_features)
                augmented_source_features = augmented_source_features.view(augmented_source_features.size(0), -1)
            elif model_name == 'resnet18':
                augmented_source_features = nn.AdaptiveAvgPool2d((1, 1))(augmented_source_features)
                augmented_source_features = augmented_source_features.view(augmented_source_features.size(0), -1)

            # normalize the features
            target_features = F.normalize(target_features, p=2, dim=1)
            augmented_target_features = F.normalize(augmented_target_features, p=2, dim=1)

            source_features = F.normalize(source_features, p=2, dim=1)
            augmented_source_features = F.normalize(augmented_source_features, p=2, dim=1)

            contrastive_loss = ContrastiveLoss(temperature=0.8)
            loss_target_con = contrastive_loss(target_features, augmented_target_features)
            loss_source_con = contrastive_loss(source_features, augmented_source_features)


            # Compute the differential loss
            alpha = args.alpha
            loss_source = criterion(source_outputs, source_labels)

            # self-supervised method (LEVEL 2 and 3)
            differential_loss = loss_source + torch.log(1+alpha*loss_source_con/loss_target_con)

            # Backward pass and optimize
            differential_loss.backward()

            # Mask the gradients to ensure only critical weights are updated
            # The update local help us to use small number of samples to retain the performance of source domain.
            for name, param in feature_extractor.named_parameters():
                if name in mask:
                    param.grad *= mask[name]

            optimizer.step()
            total_loss += differential_loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/nbatch:.2f}, Source Loss: {loss_source.item():.2f}, Target Loss: {loss_source.item():.2f}, Target Contrastive Loss: {loss_target_con.item():.2f}, Source Contrastive Loss: {loss_source_con.item():.2f}, Differential Loss: {differential_loss.item():.4f}")
        

    # Reset requires_grad for all parameters after updating
    for param in feature_extractor.parameters():
        param.requires_grad = True

    changed_weights = check_weight_update(prev_model, feature_extractor)

    logging.info(f"Changed weights: {changed_weights}")

    return feature_extractor

def get_model(model_name, number_classes):
    if model_name == 'resnet18':
        model = models.resnet18()
        remove_bn_from_model(model)
    elif model_name == 'vgg11':
        model = models.vgg11()

    # adapt the model to the new task 
    if model_name == 'resnet18':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, number_classes)
    elif model_name == 'vgg11':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, number_classes)
    return model

def get_classifier(model_name, number_classes):
    if model_name == 'vgg11':
        new_target_classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, number_classes),
        )
    elif model_name == 'resnet18':
        new_target_classifier = nn.Linear(512, number_classes)
    return new_target_classifier

def main(args):
    set_seed(args.manualSeed)
    print('==> Preparing data..')
    # Data loading code
    source_train_loader, source_test_loader, source_num_classes = prepare_source_data(args)
    target_train_loader, target_test_loader, target_num_classes = prepare_target_data(args)

    model_name = args.arch
    source_model = get_model(model_name, source_num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    source_model = source_model.to(device)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(args.resume):
        logging.info('Model checkpoint found. Loading...')
        source_model = load_model(args, source_model)
        logging.info('Model loaded successfully. Skipping training...')
    else:
        logging.info(f'Model checkpoint not found. {args.resume})')
        return

    # The target and source model share the same feature extractor
    if model_name == 'vgg11':
        feature_extractor = source_model.features  # For VGG11
        source_classifier = source_model.classifier
    elif model_name == 'resnet18':
        feature_extractor = ResNetFeatureExtractor(source_model)  # For ResNet18
        source_classifier = source_model.fc

    # The target classifier share the same architecture with the source classifier
    if model_name == 'vgg11':
        target_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, target_num_classes),
        )
        target_classifier.to(device)
    elif model_name == 'resnet18':
        # copy from source classifier
        target_classifier = nn.Linear(512, target_num_classes)
        target_classifier.to(device)

    # Evaluate the source model on the source data (Before encoderlock)
    src_loss, src_acc = evaluate_feature_extractor(feature_extractor, source_classifier, source_test_loader, criterion, device, model_name=model_name)

    new_target_classifier = get_classifier(model_name, target_num_classes)
    new_target_classifier.to(device)
    # train the target classifier on the target data
    new_target_classifier = transfer_learning(args, feature_extractor, new_target_classifier, target_train_loader, target_test_loader, criterion, device, model_name=model_name, learning_rate=1e-4, data_volume=args.volume)
    tgt_loss, tgt_acc = evaluate_feature_extractor(feature_extractor, new_target_classifier, target_test_loader, criterion, device, model_name)

    # The model can't be transfer direct to the target dataset --> assume we do not have label for the target domain.
    torch.cuda.empty_cache()
    # Now we get the pretrained feature extractor and the source and target classifier, we are going to 
    # 1. Select the critical weights to target classifier but not for source classifier
    # 2. Back propagate the differential loss to find how to update those weights 
    # 3. Update the weights of classifiers to make it adaptively transferable.
    # 4. Evaluate the final tranferability
    # 5. Save the best feature extractor, difference between the source loss and target loss should be as large as possible
    best_feature_extractor = copy.deepcopy(feature_extractor)
    best_acc_difference = src_acc-tgt_acc
    save_feature_extractor(args, best_feature_extractor)
    mask = {}

    for i in range(args.R): 
        logging.info(f"Upper Level Optimization Round: {i} \n")
        start_time = time.time()
        mask = upper_level_optimization(feature_extractor, source_classifier, source_train_loader, target_train_loader, mask, device, model_name)
        end_time = time.time()
        logging.info(f"Time for searching critical weights: {end_time-start_time} \n")

        start_time = time.time()
        feature_extractor = update_critical_weights(feature_extractor, source_classifier, new_target_classifier, mask, source_train_loader, target_train_loader, device, model_name, epochs=args.E)
        end_time = time.time()
        logging.info(f"Time for updating critical weights: {end_time-start_time} \n")

        if i%5 == 4:
            src_loss, src_acc = evaluate_feature_extractor(feature_extractor, source_classifier, source_test_loader, criterion, device, model_name=model_name)
            print("Evaluate source accuracy: ", src_acc)
            new_target_classifier = get_classifier(model_name, target_num_classes)
            new_target_classifier.to(device)
            print("Retrain Target Classifier")
            new_target_classifier = transfer_learning(args, feature_extractor, new_target_classifier, target_train_loader, target_test_loader, criterion, device, model_name=model_name, learning_rate=args.learning_rate, data_volume=args.volume)        
            tgt_loss, tgt_acc = evaluate_feature_extractor(feature_extractor, new_target_classifier, target_test_loader, criterion, device, model_name)
            print("Evaluate target accuracy: ", tgt_acc)
            # Save the encoder for this round 
            save_path = "modified_models/unsupervised/" + f'{args.arch}--{args.dataset}--{args.std_dataset}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(feature_extractor.state_dict(), save_path + f'/feature_extractor.pth')
            if src_acc - tgt_acc > best_acc_difference:
                best_acc_difference = src_acc - tgt_acc
                best_feature_extractor = copy.deepcopy(feature_extractor)
                save_feature_extractor(args, best_feature_extractor)

    logging.info("Evaluation before retraining: \n")
    print("Evaluate Source Classifier")
    src_loss, src_acc = evaluate_feature_extractor(best_feature_extractor, source_classifier, source_test_loader, criterion, device, model_name)
    print("Evaluate Target Classifier")
    tgt_loss, tgt_acc = evaluate_feature_extractor(best_feature_extractor, new_target_classifier, target_test_loader, criterion, device, model_name)

    # print("Retrain Source Classifier")
    new_source_classifier = get_classifier(model_name, target_num_classes)
    new_source_classifier.to(device)
    print("Retrain Source Classifier")
    new_source_classifier = transfer_learning(args, best_feature_extractor, new_source_classifier, source_train_loader, source_test_loader, criterion, device, model_name=model_name, learning_rate=1e-4, data_volume=args.volume)

    new_target_classifier = get_classifier(model_name, target_num_classes)
    new_target_classifier.to(device)
    print("Retrain Target Classifier")
    new_target_classifier = transfer_learning(args, best_feature_extractor, new_target_classifier, target_train_loader, target_test_loader, criterion, device, model_name=model_name, learning_rate=1e-4, data_volume=args.volume)
    
    logging.info("Retraining Evaluation: \n")

    print("Evaluate Source Classifier")
    new_src_loss, new_src_acc = evaluate_feature_extractor(best_feature_extractor, new_source_classifier, source_test_loader, criterion, device, model_name)
    
    print("Evaluate Target Classifier")
    new_tgt_loss, new_tgt_acc = evaluate_feature_extractor(best_feature_extractor, new_target_classifier, target_test_loader, criterion, device, model_name)
    torch.cuda.empty_cache()
        

if __name__ == '__main__':
    # Save logger file with date and time in filename
    args = parse_args()
    log_dir = 'logs/unsupervised-encoderlock/' + f'{args.arch}-{args.dataset}-{args.std_dataset}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(log_dir + 'training-encoderlock' + time.strftime("-%Y%m%d-%H%M%S") + '.log')
    logging.basicConfig(handlers=[file_handler], level=logging.INFO)
    # Current arguments
    logging.info(args)
    main(args)