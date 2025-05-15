import torch
import torch.nn as nn
import torch.nn.functional as F

class HHAEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)  # Output: (B, output_dim)


class DepthConditionedLoRA(nn.Module):
    def __init__(self, in_dim, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, in_dim, bias=False)
        self.depth_proj = nn.Linear(in_dim, in_dim)

    def forward(self, rgb_feat, depth_feat):
        B, N, C = rgb_feat.shape
        depth_context = self.depth_proj(depth_feat).unsqueeze(1).expand(B, N, C)
        delta = self.lora_B(self.lora_A(rgb_feat + depth_context)) * self.scaling
        return rgb_feat + delta


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, out_classes, lora_r=4):
        super().__init__()
        self.requires_patch_divisible_input = True
        self.patch_size = 14
        self.out_classes = out_classes
        self.depth_info = True

        # Load pretrained DINOv2 backbone
        self.encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.decoder = nn.Linear(self.encoder.embed_dim, out_classes)

        # HHA encoder (3-channel depth input)
        self.hha_encoder = HHAEncoder(input_channels=3, output_dim=self.encoder.embed_dim)

        # Depth-conditioned LoRA fusion
        self.fusion = DepthConditionedLoRA(in_dim=self.encoder.embed_dim, r=lora_r)
    
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.train()


    def forward(self, rgb, hha):
        # Resize inputs to match patch size
        new_height = (rgb.shape[2] // self.patch_size) * self.patch_size
        new_width = (rgb.shape[3] // self.patch_size) * self.patch_size
        rgb = F.interpolate(rgb, size=(new_height, new_width), mode='bilinear', align_corners=False)
        hha = F.interpolate(hha, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Get frozen RGB features from DINOv2
        with torch.no_grad():
            rgb_feat = self.encoder.get_intermediate_layers(rgb, n=1)[0]  # (B, N, C)

        # Get HHA embedding
        hha_feat = self.hha_encoder(hha)  # (B, C)

        # Apply LoRA fusion
        fused_feat = self.fusion(rgb_feat, hha_feat)  # (B, N, C)

        # Prepare for decoding
        B, N, C = fused_feat.shape
        H = new_height // self.patch_size
        W = new_width // self.patch_size
        fused_feat = fused_feat.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        logits = self.decoder(fused_feat.permute(0, 2, 3, 1))  # (B, H, W, C)
        logits = logits.permute(0, 3, 1, 2)  # (B, C, H, W)
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits
