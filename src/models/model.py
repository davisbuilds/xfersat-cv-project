import torch
import torch.nn as nn
from torchvision import models

def create_model(model_name='resnet18', num_classes=10, freeze_backbone=True):
    """
    Create a model with a pretrained backbone
    
    Args:
        model_name: Name of the model to use (choose from resnet18, resnet50, efficientnet_b0, efficientnet_b4)
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the backbone weights
        
    Returns:
        model: PyTorch model
    """
    # Load pretrained model
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the last fully connected layer
    if model_name.startswith('resnet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name.startswith('efficientnet'):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model