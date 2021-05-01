import random
import torch
import cv2

def select_random_frame(clip):
    clip_len = clip.shape[0] # L x H x W x C
    i = random.randrange(clip_len)
    return clip[i] # H x W x C

def select_random_frames(clip, num_frames):
    clip_len = clip.shape[0] # L x H x W x C
    step = int(clip_len / (num_frames+1))
    frames = []
    for i in range(num_frames):
        i = i*step + random.randrange(step)
        frames.append(clip[i])
    return torch.stack(frames, dim=0) # OL x H x W x C

def random_trim_video(clip, num_frames):
    clip_len = clip.shape[0] # L x H x W x C
    start = random.randrange(clip_len - num_frames)
    end = start + num_frames
    return clip[start:end] # OL x H x W x C

def to_optical_flow(clip):
    clip_len = clip.shape[0] # L x H x W x C
    flows = []
    for i in range(clip_len-1):
        before = cv2.cvtColor(clip[i], cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(clip[i+1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(torch.from_numpy(flow)) # See https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    return torch.stack(flows, dim=0) # OL x H x W x 2, where OL = 2*(L-1)