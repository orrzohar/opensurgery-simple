from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


import skimage.io
import skimage.transform
import skimage.color
import skimage

import cv2


class InferenceDataset(Dataset):
    """Inference dataset."""

    def __init__(self, root_dir, file_name, num_frames_per_clip, fps=13, num_classes=4, min_side=608, max_side=1024, csv_classes=None):
        self.root_dir    = root_dir
        self.file_name   = file_name
        self.num_classes = num_classes
        self.fps         = fps
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        
        if csv_classes is not None:
            self.labels = pd.read_csv(csv_classes, header=None, index_col=[1]).to_dict()[0]

        cap = cv2.VideoCapture(f'{self.root_dir}/{self.file_name}')
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        smallest_side = min(height, width)
        largest_side = max(width, height)

        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
            
        self.height = int(height * scale)
        self.width = int(width * scale)
        self.scale =scale

        self.Normalize =  transforms.Normalize(self.mean, self.std)
        self.Resize = transforms.Resize((self.height, self.width))

        if not cap.isOpened():
            print('Error: could not open video file')
        else:
            # Get the total number of frames in the video
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.video_length = self.num_frames / self.video_fps
            print(f'Video length: {self.video_length:.2f} seconds')
        # Release the video capture object
        cap.release()

        self.frames_per_batch = num_frames_per_clip * self.video_fps // self.fps
        self.num_frames_per_clip = num_frames_per_clip
        
        frames = np.linspace(0,self.num_frames,int(self.num_frames*self.fps/self.video_fps), dtype=int)
        self.video_timestamps = [frames[i:i+num_frames_per_clip] for i in range(0, len(frames), num_frames_per_clip)]
        self.num_clips = len(self.video_timestamps)
        print(f'video has {self.num_clips} clips, each with {num_frames_per_clip} frames')


    def __len__(self):
        return len(self.video_timestamps)-1
    
    def __getitem__(self, idx):
        original_frames = self.video_timestamps[idx]
        cap = cv2.VideoCapture(f'{self.root_dir}/{self.file_name}')
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_frames[0])
        for _ in range(original_frames[-1]-original_frames[0]+1):
            ret, frame = cap.read()
            if ret:
                frame = torch.from_numpy(frame / 255.)
                frame = self.Normalize(frame)
                frame = self.Resize(frame.permute(2, 0, 1))
                frames.append(frame)

            else:
                break

        # Release the video capture object
        cap.release()
        frames = torch.stack(frames, dim=0)
        return frames[original_frames-original_frames[0]]    
    
    def newgetitem(self, idx):
        original_frames = self.video_timestamps[idx]
        cap = cv2.VideoCapture(f'{self.root_dir}/{self.file_name}')
        f1 = original_frames[0]
        f_final = original_frames[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f1)

        frames = []
        for i in range(f1, f_final):
            ret, frame = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()

            if ret:
                frame = torch.from_numpy(frame / 255.)
                frame = self.Normalize(frame)
                frame = self.Resize(frame.permute(2, 0, 1))
                frames.append(frame)

            else:
                break

        # Release the video capture object
        cap.release()
        frames = torch.stack(frames, dim=0)
        return frames

    def num_classes(self):
        return self.num_classes


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=608, max_side=1024):

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return torch.from_numpy(new_image)



class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image):
        return (image.astype(np.float32) - self.mean) / self.std


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def collater(data):
    imgs = [s['img'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
