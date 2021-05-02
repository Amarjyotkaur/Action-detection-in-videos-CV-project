import cv2
import torch
import random
import numpy as np

from torchvision.transforms.functional import (
    to_tensor, to_pil_image, normalize, resize, scale,
    pad, crop, center_crop, resized_crop, hflip, vflip,
    perspective, five_crop, ten_crop, adjust_brightness,
    adjust_contrast, adjust_saturation, adjust_hue,
    adjust_gamma, rotate, affine, to_grayscale, erase,
)

from torchvision.transforms._functional_video import (
    crop as crop_video,
    resize as resize_video,
    resized_crop as resized_crop_video,
    center_crop as center_crop_video,
    normalize as normalize_video,
    hflip as hflip_video,
)

def select_random_frame(clip): # L x H x W x C -> H x W x C
    clip_len = clip.shape[0]
    i = random.randrange(clip_len)
    return clip[i]

def select_random_frames(clip, num_frames): # L x H x W x C -> # OL x H x W x C
    clip_len = clip.shape[0]
    if clip_len < num_frames:
        raise ValueError(f'The given clip has insufficient frames ({clip_len}/{num_frames})')
    elif clip_len == num_frames:
        return clip
    step = clip_len // (num_frames+1) 
    frames = []
    for i in range(num_frames):
        i = i*step + random.randrange(step)
        frames.append(clip[i])
    return np.stack(frames, axis=0)

def select_random_trim(clip, num_frames): # L x H x W x C -> # OL x H x W x C
    clip_len = clip.shape[0]
    if clip_len < num_frames:
        raise ValueError(f'The given clip has insufficient frames ({clip_len}/{num_frames})')
    elif clip_len == num_frames:
        return clip
    start = random.randrange(clip.shape[0] - num_frames)
    end = start + num_frames
    return clip[start:end]

def to_tensor_video(clip): # L x H x W x C -> C x L x H x W
    return torch.from_numpy(clip).float().permute(3, 0, 1, 2).divide(255.0)

def to_numpy_video(tensor): # C x L x H x W -> L x H x W x C
    return tensor.mul(255.0).permute(1, 2, 3, 0).byte().numpy()

def to_optical_flow(clip): # L x H x W x C -> OL x H x W x 2, where OL = L-1
    # See https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    clip_len = clip.shape[0]
    flows = []
    for i in range(clip_len-1):
        before = cv2.cvtColor(clip[i], cv2.COLOR_RGB2GRAY)
        after = cv2.cvtColor(clip[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
    return np.stack(flows, axis=0)