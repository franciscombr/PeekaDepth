#!/usr/bin/env python3
import argparse
import yaml
import importlib
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import math
random.seed(41)
plt.rcParams["font.family"] = "Times New Roman"

from nyuv2 import NYUv2

# ─── palette & helper ────────────────────────────────────────────────────────────
NYU13_COLORS = np.array([
    [  0,   0,   0], [ 26, 126,  73], [202,  17,  65], [ 10, 126, 207],
    [ 49,  44, 181], [171, 110,  54], [181, 182, 131], [120, 176, 119],
    [  0, 124, 127], [139,  81,  21], [ 56,  56, 106], [153,  40, 214],
    [212,  38, 221], [132, 135,  82],
], dtype=np.uint8)

def colorize_seg(seg, palette=NYU13_COLORS):
    h, w = seg.shape
    return palette[seg.reshape(-1)].reshape(h, w, 3)

# ─── model instantiation from YAML ──────────────────────────────────────────────
def load_models_from_cfg(cfg_path, device):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    models = []
    for m in cfg.get("models", []):
        mod = importlib.import_module(m["model_module"])
        Cls = getattr(mod, m["model_class"])
        init_kwargs = {}
        for opt in ("backbone", "out_classes"):
            if opt in m:
                init_kwargs[opt] = m[opt]

        model = Cls(**init_kwargs)
        ckpt = torch.load(m["checkpoint_path"], map_location=device)
        sd   = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(sd)
        model.to(device).eval()
        models.append((m["name"], model))

    return models

# ─── build inputs (RGB / depth / HHA) ────────────────────────────────────────────
def build_input_tensor(model, rgb, depth, hha):
    if getattr(model, "depth_info", True):
        # choose hha over raw depth here; swap if you prefer depth
        return torch.cat([rgb, hha], dim=1)
    else:
        return rgb

# ─── main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Visualize RGB, HHA, GT and model segmentations side-by-side"
    )
    p.add_argument(
        "-c", "--config", required=True,
        help="path to your YAML config file (e.g. models_config.yaml)"
    )
    p.add_argument(
        "-i", "--index", type=int, default=None,
        help="dataset sample index to visualize (random if not provided)"
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="optional path to save the figure (e.g. out.png)"
    )
    args = p.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load models
    models = load_models_from_cfg(args.config, device)

    # prepare dataset
    rgb_tf   = transforms.Compose([transforms.ToTensor()])
    depth_tf = transforms.ToTensor()
    seg_tf   = transforms.ToTensor()
    hha_tf   = transforms.ToTensor()

    ds = NYUv2(
        root="data/raw/NYUv2", train=False, download=True,
        rgb_transform=rgb_tf,
        depth_transform=depth_tf,
        seg_transform=seg_tf,
        hha_transform=hha_tf
    )

    # pick index
    idx = args.index if args.index is not None else random.randrange(len(ds))
    rgb, seg_gt, depth, hha = ds[idx]

    # convert originals for plotting
    rgb_np     = rgb.permute(1,2,0).numpy()
    hha_np     = hha.permute(1,2,0).numpy()
    seg_gt_np  = seg_gt.squeeze(0).numpy().astype(np.int64)
    seg_gt_rgb = colorize_seg(seg_gt_np)

    # run each model
    preds = []
    for name, model in models:
        inp = build_input_tensor(
            model,
            rgb.unsqueeze(0).to(device),
            depth.unsqueeze(0).to(device),
            hha.unsqueeze(0).to(device)
        )
        with torch.no_grad():
            out  = model(inp)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        preds.append((name, colorize_seg(pred)))

    # plot
    max_cols    = 3
    n_models    = len(preds)
    pred_rows   = math.ceil(n_models / max_cols)
    total_rows  = 1 + pred_rows   # one row for RGB/HHA/GT + pred_rows for model outputs
    fig, axes = plt.subplots(
        total_rows, max_cols,
        figsize=(3 * max_cols, 3 * total_rows),
        dpi=150,
        gridspec_kw={ 'wspace': 0.06, 'hspace':-0.25}
    )

    # Row 0: RGB | HHA | GT
    orig_titles = ["RGB Image", "HHA Depth", "Ground Truth"]
    orig_imgs   = [rgb_np, hha_np, seg_gt_rgb]
    for col in range(max_cols):
        ax = axes[0, col]
        if col < len(orig_imgs):
            ax.imshow(orig_imgs[col])
            ax.set_title(orig_titles[col], fontsize=15)
        ax.axis("off")

    # Rows 1…end: one cell per model (wrap at 3 per row)
    for idx, (name, pred_rgb) in enumerate(preds):
        row = 1 + (idx // max_cols)
        col = idx % max_cols
        ax  = axes[row, col]
        ax.imshow(pred_rgb)
        ax.set_title(name, fontsize=15)
        ax.axis("off")

    # blank out any unused “tail” cells
    total_cells = pred_rows * max_cols
    for idx in range(n_models, total_cells):
        row = 1 + (idx // max_cols)
        col = idx % max_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

