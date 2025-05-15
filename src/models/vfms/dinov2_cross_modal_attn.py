import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2DepthEncoder(nn.Module):
    def __init__(self, backbone: str, patch_size: int = 14, num_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size

        # 1) RGB backbone
        self.rgb_backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        D = self.rgb_backbone.embed_dim

        # 2) Depth → patch tokens
        self.depth_proj = nn.Conv2d(
            in_channels=3,
            out_channels=D,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 3) Cross‐modal fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True)

    def freeze(self):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = False
        self.rgb_backbone.eval()

    def unfreeze(self):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = True
        self.rgb_backbone.train()

    def forward(self, x: torch.Tensor):
        """
        x: (B,4,H,W)  channels=[R,G,B,Depth]
        returns:
          tokens: (B, N, D)
          grid:   (h, w) – number of patches in H and W
        """
        B, C, H, W = x.shape
        assert C == 6, "Expected 6 channels: RGBD"

        # make divisible by patch_size
        Hn = (H // self.patch_size) * self.patch_size
        Wn = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(Hn, Wn), mode='bilinear', align_corners=False)

        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # --- RGB tokens from DINOv2 ---
        rgb_tokens = self.rgb_backbone.get_intermediate_layers(rgb, n=1)[0]
        # shape = (B, N, D)

        # --- Depth tokens via conv ---
        d = self.depth_proj(depth)          # (B, D, h, w)
        B, D, h, w = d.shape
        depth_tokens = d.flatten(2).transpose(1, 2)  # (B, N, D)

        # --- Cross‐Attention: RGB queries depth ---
        fused, _ = self.cross_attn(
            query=rgb_tokens, 
            key=depth_tokens, 
            value=depth_tokens
        )
        rgb_tokens = rgb_tokens + fused    # residual fusion

        return rgb_tokens, (h, w)


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int,
                 patch_size: int = 14, num_heads: int = 8):
        super().__init__()
        # encoder now returns token embeddings + grid dims
        self.encoder = DINOv2DepthEncoder(backbone, patch_size, num_heads)
        D = self.encoder.rgb_backbone.embed_dim

        # decoder is an nn.Linear over tokens
        self.decoder = nn.Linear(D, out_classes)
    
    def freeze_encoder(self):
        self.encoder.freeze()

    def unfreeze_encoder(self):
        self.encoder.unfreeze()

    def forward(self, x: torch.Tensor):
        """
        x: (B,4,H,W)
        """
        B, C, H, W = x.shape
        tokens, (h, w) = self.encoder(x)      # tokens: (B, N, D)
        N, D = tokens.shape[1], tokens.shape[2]

        # per‐token classification
        logits = self.decoder(tokens)         # (B, N, out_classes)

        # reshape to spatial map
        logits = logits.transpose(1, 2).contiguous()  # (B, out_classes, N)
        logits = logits.view(B, -1, h, w)            # (B, out_classes, h, w)

        # upsample to input resolution
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

