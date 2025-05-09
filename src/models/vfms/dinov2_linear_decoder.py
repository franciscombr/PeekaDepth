import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, out_classes, freeze_encoder):
        super().__init__()
        self.requires_patch_divisible_input = True
        self.patch_size = 14
        self.depth_info = False
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        self.out_classes = out_classes
        self.linear_head = nn.Linear(self.backbone.embed_dim, out_classes)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()

    def forward(self, x):
        new_height =  (x.shape[2] // self.patch_size) * self.patch_size 
        new_width =  (x.shape[3] // self.patch_size) * self.patch_size 
        x_resized = torch.nn.functional.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners = False)
        
        features = self.backbone.get_intermediate_layers(x_resized, n=1)[0]  # Shape: (B, N, C)
        B, N, C = features.shape
        H = (new_height // self.patch_size) 
        W = (new_width // self.patch_size)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        logits = self.linear_head(features.permute(0, 2, 3, 1))  # (B, H, W, C)
        logits = logits.permute(0, 3, 1, 2)  # (B, C, H, W)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
