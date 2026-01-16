from torchvision import transforms
from torch.utils.data import DataLoader
from nyuv2 import NYUv2
import torch 
from collections import Counter
import numpy as np


def compute_mfb_weights(loader, num_classes, ignore_index=0):
    counts = Counter()
    total_pixels = 0
    for _, seg, _, _ in loader:
        tgt = seg.squeeze(1).view(-1).numpy()
        mask = (tgt != ignore_index)
        tgt = tgt[mask]
        counts.update(tgt.tolist())
        total_pixels += tgt.size

    freqs = np.array([ counts[i] for i in range(num_classes) ], dtype=np.float64)
    freqs = freqs / total_pixels
    weights =1 -  freqs / np.maximum(np.sum(freqs), 1e-6)
    return torch.from_numpy(weights).float()

def compute_samples_per_cls(loader, num_classes, ignore_index=0):
    counts = Counter()
    total_pixels = 0
    for _, seg, _, _ in loader:
        tgt = seg.squeeze(1).view(-1).numpy()
        mask = (tgt != ignore_index)
        tgt = tgt[mask]
        counts.update(tgt.tolist())
        total_pixels += tgt.size

    freqs = np.array([ counts[i] for i in range(num_classes) ], dtype=np.float64)/total_pixels
    return torch.from_numpy(freqs).float()

if __name__ == "__main__":
    # ---------- transforms ----------
    rgb_tf   = transforms.Compose([
        transforms.ToTensor()                
    ])
    depth_tf = transforms.ToTensor()         
    seg_tf   = transforms.ToTensor()         
    hha_tf = transforms.ToTensor()
    # ---------- dataset & loader ----------
    train_ds = NYUv2(
        root="data/raw/NYUv2",               
        train=True,
        download=True,                       
        rgb_transform=rgb_tf,
        depth_transform=depth_tf,
        seg_transform=seg_tf,                 
        hha_transform=hha_tf
    )
    train_ld = DataLoader(train_ds, batch_size=32,
                        shuffle=True, num_workers=8, pin_memory=True)


#    weights = compute_mfb_weights(train_ld, 14, ignore_index=0)
#    print(weights)
    print(compute_samples_per_cls(train_ds,14,None))