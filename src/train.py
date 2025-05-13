import enum
import os
import argparse
import importlib
from typing import Dict
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.metrics import compute_confusion_matrix, mean_iou, worst_class_iou, expected_calibration_error, pixel_auroc, pixel_accuracy, mean_accuracy, frequency_weighted_iou

from data.augmentations import JointAugment, AugmentedDataset

import wandb

from nyuv2 import NYUv2


def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum_steps, scheduler):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, (rgb, seg, depth, hha) in enumerate(loader):
        # Move inputs to device
        rgb = rgb.to(device)
        depth = depth.to(device)
        seg = seg.to(device)
        hha = hha.to(device)

        if getattr(model, "depth_info", True):
            inputs = torch.cat([rgb, hha], dim=1)
        else:
            inputs = rgb
        # Segmentation mask: remove channel dim to shape [B, H, W]
        targets = seg.squeeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, targets) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for rgb, seg, depth, hha in loader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            seg = seg.to(device)
            hha = hha.to(device)
            if getattr(model, "depth_info", True):
                # Concatenate RGB and depth to form 4-channel input
                inputs = torch.cat([rgb, hha], dim=1)
            
            else:
                inputs = rgb

            targets = seg.squeeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate_metrics(model, loader, device, num_classes, ignore_index=0, n_bins=10):
    """
    Runs evaluation over loader and returns a dict of segmentation metrics:
    - Pixel Accuracy
    - Mean Accuracy
    - Mean IoU
    - Frequency Weighted IoU
    - Worst Class IoU
    - Expected Calibration Error (ECE)
    - Pixel-level AUROC
    """
    model.eval()
    all_conf, all_pred, all_tgt = [], [], []
    with torch.no_grad():
        for rgb, seg, depth, hha in loader:
            rgb, depth, seg, hha= rgb.to(device), depth.to(device), seg.to(device), hha.to(device)
            inputs = torch.cat([rgb, hha], dim=1) if getattr(model, "depth_info", True) else rgb

            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)

            conf, pred = probs.max(dim=1)
            tgt = seg.squeeze(1)

            all_conf.append(conf.cpu().flatten())
            all_pred.append(pred.cpu().flatten())
            all_tgt.append(tgt.cpu().flatten())

    all_conf = torch.cat(all_conf).numpy()
    all_pred = torch.cat(all_pred).numpy()
    all_tgt  = torch.cat(all_tgt).numpy()

    # build mask of valid pixels
    valid_mask = all_tgt != ignore_index

    # confusion matrix
    cm = compute_confusion_matrix(all_pred, all_tgt, num_classes, ignore_index)

    # basic segmentation metrics
    pix_acc = pixel_accuracy(cm)
    mean_acc = mean_accuracy(cm)
    miou    = mean_iou(cm)
    fw_iou  = frequency_weighted_iou(cm)
    worst   = worst_class_iou(cm)

    # calibration & AUROC
    correctness = (all_pred == all_tgt).astype(int)
    ece = expected_calibration_error(all_conf, correctness, n_bins, ignore_mask=valid_mask)
    auc = pixel_auroc(all_conf, correctness, ignore_mask=valid_mask)

    return {
        "val_PixelAccuracy":           pix_acc,
        "val_MeanAccuracy":            mean_acc,
        "val_MeanIoU":                 miou,
        "val_FrequencyWeightedIoU":    fw_iou,
        "val_WorstClassIoU":           worst,
        "val_ECE":                     ece,
        "val_AUROC":                   auc
    }


def make_optimizer_from_cfg(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    optim_cls = getattr(torch.optim, cfg["name"])
    pg_kwargs = []
    for pg in cfg["param_groups"]:
        submod = getattr(model, pg["module"], None)
        if submod is None:
            raise ValueError(f"Model has no attribute '{pg['module']}")
        pg_kwargs.append({
            "params": submod.parameters(),
            "lr": float(pg["lr"]),
            "weight_decay": float(cfg.get("weight_decay",0.0)),
        })
    return optim_cls(pg_kwargs)

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
    grad_accum_steps = int(cfg.get('gradient_accumulation_steps',1))

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Dynamically import and instantiate model
    mod = importlib.import_module(model_module)
    ModelClass = getattr(mod, model_class)
    model = ModelClass(backbone=backbone,
                       
                       out_classes=num_classes)
    
    if freeze_encoder:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()

    model = model.to(device)

     # Data transformations
    rgb_tf = transforms.Compose([
        transforms.ToTensor()
    ]) 
    depth_tf = transforms.ToTensor()
    seg_tf = transforms.ToTensor()
    hha_tf = transforms.ToTensor()
    depth_tf = transforms.ToTensor()



    # Datasets and DataLoaders
    train_raw = NYUv2(
        root = data_root, 
        train = True,
        download = True,
        rgb_transform = rgb_tf, 
        depth_transform = depth_tf,
        seg_transform = seg_tf,
        hha_transform = hha_tf

    )
    val_ds = NYUv2(
        root = data_root, 
        train = False,
        download = True, 
        rgb_transform = rgb_tf, 
        depth_transform = depth_tf,
        seg_transform = seg_tf,
        hha_transform = hha_tf
    )

    output_size = (480, 640)
    joint_tf = JointAugment(output_size)
    train_ds = AugmentedDataset(train_raw, joint_tf)
   
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)



    # watch model to log gradients & weights
    wandb.watch(model, log="all", log_freq=50)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = make_optimizer_from_cfg(model, cfg["optimizer"])
    if cfg["lr_scheduler"]["name"] == "poly":
        def lr_lambda(step):
            return (1 - step/int(cfg["lr_scheduler"]["max_iters"]))**float(cfg["lr_scheduler"]["power"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_accum_steps, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_metrics = evaluate_metrics(model, train_loader, device, num_classes, ignore_index=0)
        val_metrics   = evaluate_metrics(model, val_loader,   device, num_classes, ignore_index=0)


        print(
           f"Epoch {epoch:03d}/{epochs:03d} | "
           f"Train Loss: {train_loss:.4f} | "
           f"PixelAcc(train): {train_metrics['val_PixelAccuracy']:.4f} | "
           f"MeanAcc(train): {train_metrics['val_MeanAccuracy']:.4f} | "
           f"mIoU(train): {train_metrics['val_MeanIoU']:.4f} | "
           f"FWIoU(train): {train_metrics['val_FrequencyWeightedIoU']:.4f} | "
           f"WorstIoU(train): {train_metrics['val_WorstClassIoU']:.4f} | "
           f"ECE(train): {train_metrics['val_ECE']:.4f} | "
           f"AUROC(train): {train_metrics['val_AUROC']:.4f} || "
           f"Val Loss: {val_loss:.4f} | "
           f"PixelAcc(val): {val_metrics['val_PixelAccuracy']:.4f} | "
           f"MeanAcc(val): {val_metrics['val_MeanAccuracy']:.4f} | "
           f"mIoU(val): {val_metrics['val_MeanIoU']:.4f} | "
           f"FWIoU(val): {val_metrics['val_FrequencyWeightedIoU']:.4f} | "
           f"WorstIoU(val): {val_metrics['val_WorstClassIoU']:.4f} | "
           f"ECE(val): {val_metrics['val_ECE']:.4f} | "
           f"AUROC(val): {val_metrics['val_AUROC']:.4f}"
        )

        wandb.log({
          "epoch":                     epoch,
          "train_loss":                train_loss,
          "val_loss":                  val_loss,

          # Expanded segmentation metrics
          "train_PixelAccuracy":           train_metrics['val_PixelAccuracy'],
          "train_MeanAccuracy":            train_metrics['val_MeanAccuracy'],
          "train_mIoU":                    train_metrics['val_MeanIoU'],
          "train_FrequencyWeightedIoU":    train_metrics['val_FrequencyWeightedIoU'],
          "train_WorstClassIoU":           train_metrics['val_WorstClassIoU'],
          "train_ECE":                     train_metrics['val_ECE'],
          "train_AUROC":                   train_metrics['val_AUROC'],

          "val_PixelAccuracy":             val_metrics['val_PixelAccuracy'],
          "val_MeanAccuracy":              val_metrics['val_MeanAccuracy'],
          "val_mIoU":                      val_metrics['val_MeanIoU'],
          "val_FrequencyWeightedIoU":      val_metrics['val_FrequencyWeightedIoU'],
          "val_WorstClassIoU":             val_metrics['val_WorstClassIoU'],
          "val_ECE":                       val_metrics['val_ECE'],
          "val_AUROC":                     val_metrics['val_AUROC'],

          "lr_encoder":               optimizer.param_groups[0]['lr'],
          "lr_decoder":               optimizer.param_groups[1]['lr']
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
