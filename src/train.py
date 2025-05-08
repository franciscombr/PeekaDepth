import os
import argparse
import importlib
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb

from nyuv2 import NYUv2


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for rgb, seg, depth in loader:
        # Move inputs to device
        rgb = rgb.to(device)
        depth = depth.to(device)
        seg = seg.to(device)

        #if getattr(model, "requires_patch_divisible_input", False):
        #    patch_size = getattr(model, "patch_size", 14)
        #    new_height =  (rgb.shape[2] // patch_size) * patch_size 
        #    new_width =  (rgb.shape[3] // patch_size) * patch_size 
        #    rgb = torch.nn.functional.interpolate(rgb, size=(new_height, new_width), mode='bilinear', align_corners = False)
        #    depth = torch.nn.functional.interpolate(depth, size=(new_height, new_width), mode='bilinear', align_corners = False)
        if getattr(model, "depth_info", True):
            # Concatenate RGB and depth to form 4-channel input
            inputs = torch.cat([rgb, depth], dim=1)
            print(model.depth_info, inputs.shape)
        else:
            inputs = rgb
        
        # Segmentation mask: remove channel dim to shape [B, H, W]
        targets = seg.squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for rgb, seg, depth in loader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            seg = seg.to(device)

            inputs = torch.cat([rgb, depth], dim=1)
            targets = seg.squeeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model on NYUv2 RGB-D data using a config file.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file with training parameters')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)


    # ── W&B init ────────────────────────────────────────────────────────────
    wandb.init(
        project="rgbd_fusion",
        entity="fmribeiro",  # replace with your W&B username/team
        config=cfg
    )
    config = wandb.config


    # Extract config parameters
    data_root      = cfg['data_root']
    model_module   = cfg['model_module']
    model_class    = cfg.get('model_class', 'UNetResNet')
    backbone       = cfg.get('backbone', 'resnet34')
    weights        = cfg.get('weights', 'models.ResNet34_Weights.DEFAULT')
    freeze_encoder = cfg.get('freeze_encoder', False)
    num_classes    = int(cfg.get('num_classes', 40))
    batch_size     = int(cfg.get('batch_size', 4))
    epochs         = int(cfg.get('epochs', 100))
    lr             = float(cfg.get('lr', 1e-3))
    weight_decay   = float(cfg.get('weight_decay', 1e-5))
    num_workers    = int(cfg.get('num_workers', 4))
    out_dir        = cfg.get('out_dir', './checkpoints')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformations
    rgb_tf = transforms.Compose([
        transforms.ToTensor()
    ]) 
    depth_tf = transforms.ToTensor()
    seg_tf = transforms.ToTensor()

    # Datasets and DataLoaders
    train_ds = NYUv2(
        root = data_root, 
        train = True,
        download = True, 
        rgb_transform = rgb_tf, 
        depth_transform = depth_tf,
        seg_transform = seg_tf
    )
    val_ds = NYUv2(
        root = data_root, 
        train = False,
        download = True, 
        rgb_transform = rgb_tf, 
        depth_transform = depth_tf,
        seg_transform = seg_tf
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Dynamically import and instantiate model
    mod = importlib.import_module(model_module)
    ModelClass = getattr(mod, model_class)
    model = ModelClass(backbone=backbone,
                       
                       out_classes=num_classes,
                       freeze_encoder=freeze_encoder)
    model = model.to(device)

    # watch model to log gradients & weights
    wandb.watch(model, log="all", log_freq=50)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # log metrics to W&B
        wandb.log({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "lr":          optimizer.param_groups[0]['lr']
        })


        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, f"{model_class}_best.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Save final model
    final_path = os.path.join(out_dir, f"{model_class}_final.pth")
    torch.save(model.state_dict(), final_path)

if __name__ == '__main__':
    main()
