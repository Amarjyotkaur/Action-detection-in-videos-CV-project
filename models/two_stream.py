import torch
import torch.nn as nn

from .finetuned_resnet import finetuned_resnet
from .temporal_cnn import temporal_CNN

class two_stream_model(nn.Module):
    '''
    The simple two-stream model, it simply takes an average on the outputs of two streams and regards it as
    the final output
    :return: The two stream model that fuses the output of spatial and temporal streams
    '''
    def __init__(self, num_classes, spatial_weights_dir=None, temporal_weights_dir=None):    
        super(two_stream_model, self).__init__()
        
        # the models of different stream
        self.spatial_stream = finetuned_resnet(num_classes, spatial_weights_dir, include_top=True)
        self.temporal_stream = temporal_CNN(num_classes, temporal_weights_dir, include_top=True)

        # freeze all weights, the two models have been trained separately
        for param in self.spatial_stream.parameters():
            param.requires_grad = False
        for param in self.temporal_stream.parameters():
            param.requires_grad = False

    def forward(self, spatial_input, temporal_input):
        spatial_output = self.spatial_stream(spatial_input)
        temporal_output = self.temporal_stream(temporal_input)
        return torch.mean(torch.stack([spatial_output, temporal_output]), dim=0)