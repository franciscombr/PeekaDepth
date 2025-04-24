import random, matplotlib.pyplot as plt, numpy as np
from nyuv2 import NYUv2
from torchvision import transforms
from torch.utils.data import DataLoader

# ---------- transforms ----------
rgb_tf   = transforms.Compose([
    transforms.ToTensor()                # keep it simple for visual checks
])
depth_tf = transforms.ToTensor()         # (uint16 PNG → float tensor)
seg_tf   = transforms.ToTensor()         # (0‑13 ids → will be cast to long)

# ---------- dataset & loader ----------
train_ds = NYUv2(
    root="data/raw/NYUv2",               # same root as before
    train=True,
    download=True,                       # first run will fetch / extract PNGs
    rgb_transform=rgb_tf,
    depth_transform=depth_tf,
    seg_transform=seg_tf                 # NEW: turn on segmentation masks
)
train_ld = DataLoader(train_ds, batch_size=32,
                      shuffle=True, num_workers=8, pin_memory=True)

# ---------- sample‑and‑show helper ----------
def show_samples(dataset, n=3):
    idxs = random.sample(range(len(dataset)), n)
    for i in idxs:
        rgb, seg, depth = dataset[i]     # order = rgb → seg → depth
        rgb_np   = rgb.permute(1, 2, 0).numpy()
        depth_np = depth.squeeze().numpy()
        seg_np   = seg.squeeze().numpy()     # int IDs in [0,13]

        # RGB ----------------------------------------------------------
        plt.figure()
        plt.title(f"Sample {i} – RGB")
        plt.imshow(rgb_np)
        plt.axis("off")

        # Depth --------------------------------------------------------
        plt.figure()
        plt.title(f"Sample {i} – Depth (m)")
        plt.imshow(depth_np)             # default viridis colormap
        plt.axis("off")

        # Segmentation -----------------------------------------------
        plt.figure()
        plt.title(f"Sample {i} – Segmentation IDs")
        plt.imshow(seg_np)               # integer mask (0=background)
        plt.axis("off")

    plt.show()

# ---------- run ----------
show_samples(train_ds, n=3)