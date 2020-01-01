import os

import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import os.path as osp
import torch

class CocoMiniDetection(Dataset):
    names = ['apple','orange']
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        with open(os.path.join(root, f'annotation_{split}.pkl'), 'rb') as f:
            targets = pickle.load(f)
        fnames = os.listdir(os.path.join(root, 'cocodata'))
        fnames.sort()
        self.samples = [
            (os.path.join(root, 'cocodata', fnames[i]), targets[i])
            for i in range(len(fnames))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
      im, gt, h, w = self.pull_item(idx)
      return im, gt

    def pull_item(self,idx):
        path, target = self.samples[idx]
        img = cv2.imread(path)
        height, width, _ = img.shape
        target = np.array(target)
        bbox = target[:, :-1]
        labels = target[:, -1]
        bbox[:, ::2] /= width
        bbox[:, ::-2] /= height
        target = np.concatenate((bbox, target), axis=-1)

        if self.transform is not None:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

if __name__ == '__main__':
    dataset = CocoMiniDetection(root = '/home/junkyu/SSD2/coco2')
    print(dataset[0])
