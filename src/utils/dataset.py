import os
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_mgrid(sidelen, vmin=-1, vmax=1):
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="xy"), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

class VideoFitting_noise(Dataset):
    def __init__(self, path,  transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform


        self.video_left, self.video_right = self.get_video_tensor()
        self.num_frames, self.ch, self.H, self.W = self.video_left.size()
        print('left', self.video_left.shape)
        # self.left = self.video_left.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        self.left = self.video_left.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.right = self.video_right.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        # self.right = self.video_right.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        self.coords = get_mgrid([self.W, self.H, self.num_frames])
        # self.coords = get_mgrid([self.H, self.W, self.num_frames])

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        left_video = []
        right_video = []

        for frame in frames:
            img = Image.open(os.path.join(self.path, frame))
            print('path is', os.path.join(self.path, frame))
            # img = img.convert('L')
            img = self.transform(img)

            if 'left' in frame:
                left_video.append(img)
            elif 'right' in frame:
                right_video.append(img)

        left_video_tensor = torch.stack(left_video, 0) if left_video else None
        right_video_tensor = torch.stack(right_video, 0) if right_video else None

        return left_video_tensor, right_video_tensor

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.left, self.right