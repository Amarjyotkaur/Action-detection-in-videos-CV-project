'''
Implementing LRCN structure
References:
    http://cs231n.stanford.edu/reports2016/221_Report.pdf
    https://arxiv.org/pdf/1411.4389v3.pdf
'''
import torch
import torch.nn as nn
import os

class CNN(nn.Module):
    def __init__(self, num_classes, weights_dir=None, include_top=True):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            self.conv_block(3, 64, 7, 1, 0, batch_norm=True, max_pool=True),
            self.conv_block(64, 96, 5, 1, 0, batch_norm=True, max_pool=True),
            self.conv_block(96, 128, 3, 1, 0, batch_norm=False, max_pool=False),
            self.conv_block(128, 128, 3, 1, 0, batch_norm=False, max_pool=False),
            self.conv_block(128, 196, 3, 1, 0, batch_norm=False, max_pool=True),
        )
        
        fc_layers = [self.fc_block(196*25*25, 320, dropout=0.5)]
        if include_top:
            fc_layers.append(nn.Linear(320, num_classes))
            fc_layers.append(nn.Softmax(dim=1))
        self.fc_layers = nn.Sequential(*fc_layers)
    
        if weights_dir is not None and os.path.exists(weights_dir):
            self.load_state_dict(torch.load(weights_dir))
    
    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, batch_norm=False, max_pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    
    def fc_block(self, in_features, out_features, bias=True, dropout=None):
        layers = [
            nn.Linear(in_features, out_features, bias),
            nn.ReLU(),
        ]
        if isinstance(dropout, float) and 0 < dropout < 1:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x