import torch
import torch.nn as nn
import torchvision.models as models
import os

def finetuned_resnet(num_classes, include_top=False):
    # Load pretrained ResNet50 model
    model = models.resnet50(pretrained=True)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final fully-connected layer
    fc_layers = [
        nn.Linear(model.fc.in_features, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
    ]
    
    if include_top:
        fc_layers.append(nn.Linear(1024, num_classes))
        fc_layers.append(nn.Softmax(dim=1))
    
    model.fc = nn.Sequential(*fc_layers)
    
    return model