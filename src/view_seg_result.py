import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from nyuv2 import NYUv2
from data.augmentations import JointAugment, AugmentedDataset
from models.vfms.dinov2_cross_modal_attn import DINOv2SegmentationModel
checkpoint_path = "./results/checkpoints/models.vfms.dinov2_cross_modal_attn_final.pth"
# ------------------------------------------------------------------
# 13-class palette & names
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
# evaluate helper (just to show how inputs are built)
def build_inputs(model, rgb, depth, hha, depth_rep):
    """
    Returns the tensor you should feed to model(inputs).
    - If model.depth_info is True: concat(rgb + chosen depth rep)
    - Else: rgb only
    """
    if getattr(model, "depth_info", True):
        if depth_rep == 'hha':
            return torch.cat([rgb, hha], dim=1)
        else:
            return torch.cat([rgb, depth], dim=1)
    else:
        return rgb

# ------------------------------------------------------------------
# transforms & dataset loader
rgb_tf   = transforms.Compose([transforms.ToTensor()])
depth_tf = transforms.ToTensor()
seg_tf   = transforms.ToTensor()
hha_tf   = transforms.ToTensor()

dataset = NYUv2(
    root="data/raw/NYUv2",
    train=False,
    download=True,
    rgb_transform=rgb_tf,
    depth_transform=depth_tf,
    seg_transform=seg_tf,
    hha_transform=hha_tf
)
loader = DataLoader(dataset, batch_size=1, shuffle=False,
                    num_workers=4, pin_memory=True)

# ------------------------------------------------------------------
# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DINOv2SegmentationModel(backbone="dinov2_vitl14",out_classes=14)            
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device).eval()

# ------------------------------------------------------------------
def visualize_predictions(model, dataset, idxs=None,
                          depth_rep='hha', device='cuda',
                          save_path=None, dpi=200):
    """
    Shows for each idx: [RGB | GT seg | Predicted seg | HHA Depth]
    depth_rep: 'hha' or 'depth' (matches your evaluate logic)
    """
    if idxs is None:
        idxs = random.sample(range(len(dataset)), 2)

    fig = plt.figure(figsize=(12, 4 * len(idxs)), dpi=dpi)
    gs  = fig.add_gridspec(len(idxs), 4, wspace=0.05, hspace=0.05)
    titles = ['RGB', 'Ground Truth', 'Predicted', 'HHA Depth']

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            rgb, seg_gt, depth, hha = dataset[idx]
            rgb, depth, seg_gt, hha = rgb.to(device), depth.to(device), seg_gt.to(device), hha.to(device)

            # build the actual model input
            inp = build_inputs(model, rgb.unsqueeze(0), depth.unsqueeze(0), hha.unsqueeze(0), depth_rep)

            # forward + argmax
            logits = model(inp)
            pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            # back to CPU for plotting
            rgb_np     = rgb.cpu().permute(1,2,0).numpy()
            hha_np     = hha.cpu().permute(1,2,0).numpy()
            seg_gt_np  = seg_gt.squeeze(0).cpu().numpy().astype(np.int64)
            seg_gt_rgb = colorize_seg(seg_gt_np)
            pred_rgb   = colorize_seg(pred)

            for col, img in enumerate([rgb_np, hha_np,seg_gt_rgb, pred_rgb ]):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img if col!=2 else img)  # all RGB here; depth handled if needed
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col], fontsize=10)

    ## legend
    #ax_leg = fig.add_subplot(gs[:, -1])
    #ax_leg.axis('off')
    #box_sz = 0.08; cols = 2
    #rows   = int(np.ceil((len(NYU13_NAMES)-1)/cols))
    #for k, name in enumerate(NYU13_NAMES[1:]):
    #    col = k // rows; row = k % rows
    #    x = col * 0.48; y = 1 - (row+1) * box_sz*1.3
    #    ax_leg.add_patch(mpatches.Rectangle((x,y), box_sz, box_sz,
    #                                       facecolor=NYU13_COLORS[k+1]/255,
    #                                       transform=ax_leg.transAxes))
    #    ax_leg.text(x+box_sz*1.3, y+box_sz/2, name,
    #                transform=ax_leg.transAxes, va='center', fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------
if __name__ == "__main__":
    # e.g. visualize samples 5 and 17, using HHA depth
    visualize_predictions(model, dataset, idxs=[1],device=device,
                          depth_rep='hha',
                          save_path="pred_vs_gt.png")

