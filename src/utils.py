import cv2
import torch

def read_clip(clip_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame.any():
            frames.append(torch.from_numpy(frame))
    cap.release()
    return torch.stack(frames, dim=0) # L x H x W x C

def calculate_statistics(root):
    clips = []
    for root, _, files in os.walk(root):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == '.webm':
                clip = read_clip(os.path.join(root, file))
                clips.append(clip)
    clips = torch.cat(clips, dim=0) # ΣL x H x W x C
    mean = clips.mean(dim=(0, 1, 2)) # C
    std = clips.std(dim=(0, 1, 2)) # C
    return mean, std