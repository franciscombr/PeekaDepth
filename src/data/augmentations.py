import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

class JointAugment:
    def __init__(
        self,
        output_size,
        flip_prob=0.5,
        scale_range=(0.5, 2.0),
        color_jitter_params=dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ):
        self.output_size = output_size
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.color_jitter = transforms.ColorJitter(**color_jitter_params)

    def __call__(self, rgb, depth, seg):
        # 1) Flip
        if random.random() < self.flip_prob:
            rgb, depth, seg = TF.hflip(rgb), TF.hflip(depth), TF.hflip(seg)

        # 2) Color‐jitter RGB only
        rgb = self.color_jitter(rgb)

        # 3) Scale
        scale = random.uniform(*self.scale_range)
        # guard for both PIL.Image and torch.Tensor
        if hasattr(rgb, "shape"):
            # rgb is Tensor in C×H×W
            _, h, w = rgb.shape
        else:
            # rgb is PIL Image
            w, h = rgb.size
        new_h = int(h * scale)
        new_w = int(w * scale)

        rgb   = TF.resize(rgb,   (new_h, new_w), interpolation=Image.BILINEAR)
        depth = TF.resize(depth, (new_h, new_w), interpolation=Image.NEAREST)
        seg   = TF.resize(seg,   (new_h, new_w), interpolation=Image.NEAREST)

        # 4) Pad to at least output_size
        pad_h = max(self.output_size[0] - new_h, 0)
        pad_w = max(self.output_size[1] - new_w, 0)
        if pad_h or pad_w:
            rgb   = TF.pad(rgb, (0,0,pad_w,pad_h), fill=0)
            depth = TF.pad(depth, (0,0,pad_w,pad_h), fill=0)
            seg   = TF.pad(seg, (0,0,pad_w,pad_h), fill=0)

        # 5) Random crop
        i, j, h, w = transforms.RandomCrop.get_params(rgb, self.output_size)
        rgb   = TF.crop(rgb, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)
        seg   = TF.crop(seg, i, j, h, w)

        # 6) ToTensor; keep seg as LongTensor mask
        rgb   = TF.to_tensor(rgb)
        depth = TF.to_tensor(depth)
        seg   = torch.from_numpy(np.array(seg)).long()
        return rgb, depth, seg

class AugmentedDataset(Dataset):
    def __init__(self, base_ds, joint_tf):
        self.base = base_ds
        self.joint_tf = joint_tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # NYUv2 returns (rgb: PIL, seg: PIL, depth: PIL) or vice-versa
        rgb, seg, depth = self.base[idx]
        return self.joint_tf(rgb, depth, seg)
