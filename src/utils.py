import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from .functional import to_tensor_video

def read_clip(clip_path):
    frames = []
    cap = cv2.VideoCapture(clip_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame.any():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    if frames:
        return np.stack(frames, axis=0) # L x H x W x C
    else:
        raise ValueError(f"'{clip_path}' does not contain any frames")

def calculate_statistics(root):
    # See https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch
    nimages = 0
    mean = torch.zeros(3)
    var = torch.zeros(3)
    get_mean = lambda: tuple((mean / (nimages+1e-9)).data.tolist())
    get_std = lambda: tuple((var / (nimages+1e-9)).sqrt().data.tolist())
    fmt = lambda tup: str(tuple(round(v, 4) for v in tup))
    progress = tqdm(unit=' clips', postfix=f'mean={fmt(get_mean())}, std={fmt(get_std())}')
    for root, _, files in os.walk(root):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == '.webm':
                clip_path = os.path.join(root, file)
                clip = to_tensor_video(read_clip(clip_path)) # C x L x H x W
                nimages += clip.size(1)
                mean += clip.mean(dim=(2, 3)).sum(dim=1)
                var += clip.var(dim=(2, 3)).sum(dim=1)
                progress.update(1)
                progress.set_postfix_str(f'mean={fmt(get_mean())}, std={fmt(get_std())}')
    return get_mean(), get_std()