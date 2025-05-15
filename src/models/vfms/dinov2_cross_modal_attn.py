import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2DepthFusionSegModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int, 
                 num_heads: int = 8, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size

        # RGB encoder (pretrained DINOv2)
        self.rgb_encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        D = self.rgb_encoder.embed_dim

        # Depth projection: single‐channel → D‐dim patch tokens
        self.depth_proj = nn.Conv2d(
            in_channels=3, 
            out_channels=D, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        # Cross‐modal attention: RGB tokens query depth tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True)

        # Final linear decoder (per‐patch classifier)
        self.decoder = nn.Linear(D, out_classes)

    def freeze_encoder(self):
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
        self.rgb_encoder.eval()

    def unfreeze_encoder(self):
        for p in self.rgb_encoder.parameters():
            p.requires_grad = True
        self.rgb_encoder.train()

    def forward(self, x: torch.Tensor):
        """
        x: (B, 4, H, W) where channels = [R,G,B,Depth]
        """
        B, C, H, W = x.shape
        assert C == 6, "Input must have 4 channels: RGBD"

        # Resize so H,W divisible by patch_size
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # Split RGB / Depth
        rgb = x[:, :3, :, :]         # (B,3,H',W')
        depth = x[:, 3:, :, :]       # (B,1,H',W')

        # --- RGB branch: get last‐layer tokens from DINOv2 ---
        rgb_tokens = self.rgb_encoder.get_intermediate_layers(rgb, n=1)[0]
        # shape = (B, N, D) where N = num_patches

        # --- Depth branch: conv → flatten to tokens ---
        d = self.depth_proj(depth)  # (B, D, H'/ps, W'/ps)
        B, D, h, w = d.shape
        depth_tokens = d.flatten(2).transpose(1, 2)  # → (B, N, D)

        # --- Cross‐Modal Attention: RGB queries depth ---
        # MultiheadAttention expects (B, N, D) if batch_first=True
        fused_tokens, _ = self.cross_attn(
            query=rgb_tokens,    # (B, N, D)
            key=depth_tokens,    # (B, N, D)
            value=depth_tokens   # (B, N, D)
        )
        # Residual fusion
        rgb_tokens = rgb_tokens + fused_tokens       # (B, N, D)

        # --- Decode per‐token and reshape to map ---
        logits = self.decoder(rgb_tokens)            # (B, N, out_classes)
        # back to (B, out_classes, h, w)
        logits = logits.transpose(1, 2)              # (B, out_classes, N)
        logits = logits.view(B, self.decoder.out_features, h, w)
        # Upsample to original H,W
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits
