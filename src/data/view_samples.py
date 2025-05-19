import random, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import random, matplotlib.pyplot as plt, numpy as np
from nyuv2 import NYUv2
from torchvision import transforms
from torch.utils.data import DataLoader
random.seed(42)
plt.rcParams["font.family"] = "Times New Roman"
from augmentations import JointAugment, AugmentedDataset
import torch
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


# ------------------------------------------------------------------
# 13-class NYUv2 palette  (0 -> background)
NYU13_COLORS = np.array([
    [  0,   0,   0],   
    [ 26, 126,  73],   
    [202,  17,  65],   
    [ 10, 126, 207],   
    [ 49,  44, 181],   
    [171, 110,  54],   
    [181, 182, 131],   
    [120, 176, 119],   
    [  0, 124, 127],   
    [139,  81,  21],   
    [ 56,  56, 106],   
    [153,  40, 214],   
    [212,  38, 221],   
    [132, 135,  82],   
], dtype=np.uint8)

NYU13_NAMES = [
    'Unlabeled','Bed','Books','Ceiling','Chair','Floor','Furniture',
    'Object','Painting','Sofa','Table','TV','Wall','Window'
]

def colorize_seg(seg, palette=NYU13_COLORS):
    h, w = seg.shape
    return palette[seg.reshape(-1)].reshape(h, w, 3)

# ------------------------------------------------------------------
def make_figure(dataset, idxs=None, save_path=None, dpi=300):
    if idxs is None:
        idxs = random.sample(range(len(dataset)), 2)
    depths_for_range = []
    
    for i in idxs:
        _, _, depth, _ = dataset[i]
        d = depth.squeeze().numpy()
        depths_for_range.append(d[d > 0])       
    all_depths = np.concatenate(depths_for_range)
    vmin, vmax = np.percentile(all_depths, [1, 99]) 
    fig = plt.figure(figsize=(10, 4), dpi=dpi)
    gs  = fig.add_gridspec(2, 5, width_ratios=[1,1,1,1,0.7],
                           wspace=0.06, hspace=0.05)

    titles = ['Image', 'Ground Truth', 'Depth', 'HHA Depth']

    for r, i in enumerate(idxs):
        rgb, seg, depth, hha = dataset[i]
        rgb_np   = rgb.permute(1, 2, 0).numpy()
        depth_np = depth.squeeze().numpy()
        hha_np   = hha.permute(1, 2, 0).numpy()
        seg_rgb  = colorize_seg(seg.squeeze().numpy().astype(np.int64))

        for c, img in enumerate([rgb_np, seg_rgb, depth_np, hha_np]):
            ax = fig.add_subplot(gs[r, c])
            if c == 2:           
                ax.imshow(abs(img), cmap='magma', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9 )

    ax_leg = fig.add_subplot(gs[:, 4])
    ax_leg.axis('off')

    box_sz = 0.09
    cols   = 2
    rows   = int(np.ceil((len(NYU13_NAMES)-1)/cols))
    for k, name in enumerate(NYU13_NAMES[1:]):       # 1..13
        col = k // rows
        row = k %  rows
        x   = col * 0.53
        y   = 1 - (row+1) * box_sz*1.3
        ax_leg.add_patch(
            mpatches.Rectangle((x, y), box_sz, box_sz,
                               facecolor=NYU13_COLORS[k+1]/255,
                               transform=ax_leg.transAxes, clip_on=False)
        )
        ax_leg.text(x + box_sz*1.4, y + box_sz/2, name,
                     transform=ax_leg.transAxes, va='center', fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def figure_aug_examples(base_ds, joint_tf,
                        idxs=None,           # list/tuple of dataset indices
                        n_aug=3,
                        save_path=None,
                        dpi=300,
                        seed=123):
    """
    Show Original + N augmentations for each index in `idxs`
    Layout  :  (n_aug+1 rows) × 4 columns  per index
    Columns :  [RGB | GT | Depth | HHA]
    """
    import random, numpy as np, matplotlib.pyplot as plt, torch

    random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------
    if idxs is None:
        idxs = random.sample(range(len(base_ds)), 1)   # ← default ONE index
    n_idx   = len(idxs)                                 # 1, 2, ...
    rows    = n_aug + 1
    cols    = 4

    # figure size scales with number of indices
    fig_w   = cols * 2.2 * n_idx
    fig_h   = rows * 2.0
    fig     = plt.figure(figsize=(fig_w, fig_h), dpi=dpi,
                          constrained_layout=True)
    gs      = fig.add_gridspec(rows*2, cols*n_idx,
                               wspace=0.05, hspace=0.05)

    titles  = ["RGB", "Ground Truth", "Depth", "HHA Depth"]

    for block, idx in enumerate(idxs):          # loop over dataset indices
        rgb0, seg0, depth0, hha0 = base_ds[idx]
        variants = [(rgb0, seg0, depth0, hha0)]          # original
        for _ in range(n_aug):                           # augmentations
            variants.append(joint_tf(rgb0, seg0, depth0, hha0))

        # depth scale for this block
        depths = np.concatenate([v[2].squeeze().numpy().ravel()
                                 for v in variants if v[2].max() > 0])
        vmin, vmax = np.percentile(depths[depths > 0], [1, 99])

        for r, (rgb, seg, depth, hha) in enumerate(variants):
            rgb_np   = rgb.permute(1,2,0).numpy() if isinstance(rgb, torch.Tensor) else np.array(rgb)
            hha_np   = hha.permute(1,2,0).numpy() if isinstance(hha, torch.Tensor) else np.array(hha)
            depth_np = depth.squeeze().numpy()     if isinstance(depth, torch.Tensor) else np.array(depth)
            seg_np   = seg.squeeze().numpy()       if isinstance(seg, torch.Tensor) else np.array(seg)
            seg_rgb  = colorize_seg(seg_np.astype(np.int64))

            imgs = [rgb_np, seg_rgb, abs(depth_np), hha_np]
            for c, img in enumerate(imgs):
                ax = fig.add_subplot(gs[r*2:(r+1)*2, block*cols + c])
                if c == 2:                                # depth
                    masked = np.ma.masked_where(img == 0, img)
                    im = ax.imshow(masked, cmap="magma",
                                   vmin=vmin, vmax=vmax)
                    #im.set_bad(alpha=0)                   # transparent holes
                    # draw colour-bar once (only for 1st block & 1st row)
                    if (block == 0) and (r == 0):
                        ...
                        #cax = fig.add_axes([0.25, 0.015, 0.50, 0.02])
                        #cb  = fig.colorbar(im, cax=cax, orientation='horizontal')
                        #cb.set_label("Depth (m)", size=8)
                        #cb.ax.tick_params(labelsize=7)
                else:
                    ax.imshow(img)
                ax.axis("off")
                if r == 0:
                    ax.set_title(titles[c], fontsize=9)

            # annotate the row on the left of each block
            row_lbl = "Original" if r == 0 else f"Aug {r}"
            y_ax = fig.add_subplot(gs[r*2:(r+1)*2, block*cols])
            y_ax.set_ylabel(row_lbl, fontsize=9, rotation=0, labelpad=30)
            y_ax.set_xticks([]); y_ax.set_yticks([]); y_ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------------
# >>> use exactly as before:
make_figure(train_ds, save_path='./paper_samples.png')
aug_tf   = JointAugment(output_size=(480, 640))
aug_ds   = AugmentedDataset(train_ds, aug_tf)
figure_aug_examples(train_ds, aug_tf, save_path="./aug_examples.png", seed=42, n_aug=2)