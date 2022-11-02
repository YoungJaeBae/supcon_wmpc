import torch
import os
import numpy as np
import torch.utils.data as data


class WM811K(data.Dataset):
    def __init__(self, root, df, transform=None, pretrained: bool = False):
        super(WM811K, self).__init__()
        self.root = root + "/wafermaps"
        self.WM_frame = df
        self.transform = transform
        self.is_pretrained = pretrained

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.WM_frame.npyPath[idx])
        img = np.load(img_path)
        if self.is_pretrained:
            img = np.repeat(img, 3, axis=2)
        label = self.WM_frame.failureNum[idx]
        target = torch.tensor(label)

        """
        transform returns a pair of positive samples
        """
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.WM_frame)


class MixedWM38(data.Dataset):
    def __init__(self, root, df, transform=None):
        super(MixedWM38, self).__init__()
        self.wafer_root = root + "/wafermaps"
        self.label_root = root + "/labels"
        self.WM_frame = df
        self.transform = transform

    def __getitem__(self, idx):

        wafer_path = os.path.join(self.wafer_root, self.WM_frame.wafernpyPath[idx])
        wafer = np.load(wafer_path).astype(np.float32)

        label = os.path.join(self.label_root, self.WM_frame.labelnpyPath[idx])
        target = torch.tensor(np.load(label), dtype=torch.float32)

        """
        transform returns a pair of positive samples
        """

        if self.transform is not None:
            wafer = self.transform(wafer)
            wafer = wafer

        return wafer, target

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.WM_frame)
