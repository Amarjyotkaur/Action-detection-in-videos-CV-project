import torch
import torch.nn as nn
import os

class temporal_CNN(nn.Module):
    '''
    The CNN for optical flow input.
    Since optical flow is not a common image, we cannot finetune pre-trained ResNet (The weights trained on imagenet is
    for images and thus is meaningless for optical flow)
    '''
    def __init__(self, num_classes, weights_dir=None, include_top=True):
        '''
        :param input_shape: the shape of optical flow input
        :param classes: number of classes
        :return:
        '''
        super(temporal_CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            self.conv_block(18, 96, 7, 2, 0, max_pool=True),
            self.conv_block(96, 256, 5, 2, 0, max_pool=True),
            self.conv_block(256, 512, 3, 1, 1, max_pool=False),
            self.conv_block(512, 512, 3, 1, 1, max_pool=False),
            self.conv_block(512, 512, 3, 1, 1, max_pool=True),
        )
        
        fc_layers = [
            self.fc_block(512*6*6, 4096, dropout=0.9),
            self.fc_block(4096, 2048, dropout=0.9),
        ]
        if include_top:
            fc_layers.append(nn.Linear(2048, num_classes))
            fc_layers.append(nn.Softmax(dim=1))
        self.fc_layers = nn.Sequential(*fc_layers)
        
        if weights_dir is not None and os.path.exists(weights_dir):
            self.load_state_dict(torch.load(weights_dir))
    
    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, max_pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
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
        if len(x.shape) == 5: # B x C X L x H x W
            x = x.view(x.shape[0], -1, *x.shape[3:]) # B x C*L x H x W
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x