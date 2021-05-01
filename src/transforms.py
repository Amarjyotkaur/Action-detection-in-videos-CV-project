import random
import torch

from torchvision.transforms._transforms_video import (
    RandomCropVideo,
    RandomResizedCropVideo,
    CenterCropVideo,
    NormalizeVideo,
    ToTensorVideo,
    RandomHorizontalFlipVideo,
)

from . import functional as F

class RandomSelectFrame(object):
    def __call__(self, clip):
        return F.select_random_frame(clip)

class RandomSelectFrames(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
    
    def __call__(self, clip):
        return F.select_random_frames(clip, self.num_frames)

class RandomTrimVideo(object):
    def __init__(self, num_frames):
        self.num_frames = num_frames
    
    def __call__(self, clip):
        return F.random_trim_video(clip, self.num_frames)

class ToOpticalFlow(object):
    def __call__(self, clip):
        return F.to_optical_flow(clip)