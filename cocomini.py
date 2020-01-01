# import os
#
# import numpy as np
# import cv2
# from torch.utils.data import Dataset
# import pickle
# import os.path as osp
# import torch
#
# class CocoMiniDetection(Dataset):
#     names = ['apple','orange']
#     def __init__(self, root, split='train', transform=None):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         with open(os.path.join(root, f'annotations_{split}.pkl'), 'rb') as f:
#             targets = pickle.load(f)
#
#         fnames = os.listdir(os.path.join(root, split))
#         fnames.sort()
#         self.samples = [
#             (os.path.join(root, split, fnames[i]), targets[i])
#             for i in range(len(fnames))
#         ]
#
#         self.ids = list()
#         name = ['apple','orange']
#         for i in name:
#             for line in open(osp.join('/home/junkyu/SSD2/class/', i + '.txt')):
#                 self.ids.append(('/home/junkyu/SSD2/class/', line.strip()))
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#       im, gt, h, w = self.pull_item(idx)
#       return im, gt
#
#     def pull_item(self,idx):
#         path, target = self.samples[idx]
#         img = cv2.imread(path)
#         height, width, _ = img.shape
#         target = np.array(target)
#         bbox = target[:, :-1]
#         labels = target[:, -1]
#
#         bbox[:, ::2] /= width
#         bbox[:, ::-2] /= height
#         target = np.concatenate((bbox, target), axis=-1)
#
#         if self.transform is not None:
#             img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
#             # to rgb
#             img = img[:, :, (2, 1, 0)]
#             # img = img.transpose(2, 0, 1)
#             target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
#         return torch.from_numpy(img).permute(2, 0, 1), target, height, width


import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pickle
COCO_ROOT = osp.join('/home/junkyu/SSD2/coco2')
IMAGES = 'images'
ANNOTATIONS = 'annotation'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('apple', 'banana','broccoli')





class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = [1,2,3]

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            bbox = obj[0:4]
            label_idx = self.label_map[int(obj[8])] - 1
            final_box = list(np.array(bbox)/scale)
            final_box.append(label_idx)
            res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, transform=None,split = 'train',
                 target_transform=COCOAnnotationTransform()):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(root, f'annotation_{split}.pkl'), 'rb') as f:
            targets = pickle.load(f)

        fnames = os.listdir(os.path.join(root, split))
        fnames.sort()
        self.samples = [
            (os.path.join(root, split, fnames[i]), targets[i])
            for i in range(len(fnames))
        ]
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.samples)

    def pull_item(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        path, target = self.samples[idx]
        img = cv2.imread(path)
        height, width, _ = img.shape
        target = np.array(target)
        bbox = target[:, :-1]
        labels = target[:, -1]

        bbox[:, ::2] /= width
        bbox[:, ::-2] /= height
        target = np.concatenate((bbox, target), axis=-1)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
