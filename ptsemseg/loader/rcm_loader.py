import os
import collections
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate
from builtins import input


class RCMLoader(data.Dataset):
    def __init__(
        self,
        root,
        split='train_1',
        is_transform=False,
        img_size=None,
        augmentations=None,
        img_norm=True,
    ):
        self.root = os.path.expanduser(root)
        self.splt = split
        self.img_size = [360, 480]
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([0])
        self.n_classes = 6
        self.files = collections.defaultdict(list)

        file_path = os.path.join(self.root, '{}.txt'.format(split))
        with open(file_path) as fp:
            self.files = [a.strip('\n')+'.png' for a in fp.readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = os.path.join(self.root, "Dataset", img_name)
        lbl_path = os.path.join(self.root, "Mask", img_name)

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        # ignore_label for this framework is 250
        lbl[lbl == 7] = 250

        return img, lbl

    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        no_lesion = [128, 128, 128]
        artifact = [128, 0, 0]
        ring = [192, 192, 128]
        nest = [128, 64, 128]
        mesh = [60, 40, 222]
        aspecific = [128, 128, 0]
        unlabeled = [0, 0, 0]

        label_colours = np.array(
            [
                no_lesion,
                artifact,
                ring,
                nest,
                mesh,
                aspecific,
                unlabeled
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        #change ignore_label back to 6
        r[r == 250] = 6
        g[b == 250] = 6
        b[b == 250] = 6

        for l in range(0, self.n_classes+1):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    local_path = "/home/alican/Datasets/NCI-Melissa-Italy_1"
    augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])

    dst = RCMLoader(local_path, is_transform=True, augmentations=augmentations, split='train_1')
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        imgs = np.squeeze(imgs)
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
