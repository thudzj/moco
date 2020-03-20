from __future__ import print_function

import torch
import torchvision.datasets as datasets
import numpy as np


class ImageFolderInstance(datasets.ImageFolder):
    """Folder dataset which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, two_crop=False):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target

class SelectedImageFolderInstance(datasets.ImageFolder):

    def __init__(self, root, k, data_seed=0, transform=None, target_transform=None):
        super(SelectedImageFolderInstance, self).__init__(root, transform, target_transform)

        rng = np.random.RandomState(data_seed)

        tmp_targets = np.array(self.targets)
        samples, targets, idx = [], [], []
        for i in range(len(self.classes)):
            idx_i = np.nonzero(tmp_targets == i)[0]
            idx_i = rng.choice(idx_i, k, replace=False)
            assert(len(np.unique(idx_i)) == k)
            assert(len(idx_i) == k)
            for ii, j in enumerate(idx_i):
                samples.append(self.samples[j])
                targets.append(self.targets[j])
                assert(self.targets[j] == i)
                idx.append(i*k + ii)
        del self.samples, self.targets
        self.samples = samples
        self.targets = targets
        self.idx = idx

        for i, (s, t) in enumerate(self.samples):
            assert (t == self.targets[i])
        for i in range(len(self.classes)):
            assert ((np.array(self.targets) == i).sum() == k)
        for i in range(len(self.classes)*k):
            assert(self.idx[i]//k == self.targets[i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        id = self.idx[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, id


class KShotImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, k, data_seed=0, transform=None, target_transform=None, two_crop=False):
        super(KShotImageFolderInstance, self).__init__(root, transform, target_transform)
        self.two_crop = two_crop

        rng = np.random.RandomState(data_seed)

        self.is_available = [0,] * len(self.targets)
        self.idx = [-1] * len(self.targets)
        cur_dix = 0
        tmp_targets = np.array(self.targets)
        for i in range(len(self.classes)):
            idx_i = np.nonzero(tmp_targets == i)[0]
            idx_i = rng.choice(idx_i, k, replace=False)
            assert(len(np.unique(idx_i)) == k)
            assert(len(idx_i) == k)
            for j in idx_i:
                self.is_available[j] = 1
                self.idx[j] = cur_dix
                cur_dix += 1

        for i in range(len(self.classes)):
            if self.is_available[i] == 1:
                assert(self.idx[i]//k == self.targets[i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        is_avail, id = self.is_available[index], self.idx[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        return img, target, is_avail, id
