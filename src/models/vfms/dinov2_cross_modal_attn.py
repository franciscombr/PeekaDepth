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

    def forward(self, x: torch.Tensor):
        """
        x: (B,6,H,W)  channels=[R,G,B,Depth_x3]
        returns:
          tokens: (B, N, D)
          grid:   (h, w) – number of patches in H and W
        """
        B, C, H, W = x.shape
        assert C == 6, "Expected 6 channels: RGB + 3 depth channels"
        Hn = (H // self.patch_size) * self.patch_size
        Wn = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(Hn, Wn), mode='bilinear', align_corners=False)

        rgb   = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # RGB tokens
        rgb_tokens = self.rgb_backbone.get_intermediate_layers(rgb, n=1)[0]  # (B, N, D)

        # Depth tokens
        d = self.depth_proj(depth)                 # (B, D, h, w)
        B, D, h, w = d.shape
        depth_tokens = d.flatten(2).transpose(1, 2) # (B, N, D)

        # Cross‐Attention fusion
        fused, _ = self.cross_attn(
            query=rgb_tokens,
            key=depth_tokens,
            value=depth_tokens
        )
        rgb_tokens = rgb_tokens + fused            # (B, N, D)

        return rgb_tokens, (h, w)


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int,
                 patch_size: int = 14, num_heads: int = 8):
        super().__init__()
        self.encoder = DINOv2DepthEncoder(backbone, patch_size, num_heads)
        D = self.encoder.rgb_backbone.embed_dim
        self.decoder = nn.Linear(D, out_classes)

    def _set_component_trainable(self, component: str, trainable: bool):
        """
        component: one of {"rgb_encoder", "depth_encoder", "decoder"}
        trainable: True to unfreeze, False to freeze
        """
        if component == "rgb_encoder":
            modules = [self.encoder.rgb_backbone]
        elif component == "depth_encoder":
            # include both depth projection and the fusion layer
            modules = [self.encoder.depth_proj, self.encoder.cross_attn]
        elif component == "decoder":
            modules = [self.decoder]
        else:
            raise ValueError(f"Unknown component '{component}'")

        for module in modules:
            # toggle grad
            for p in module.parameters():
                p.requires_grad = trainable
            # toggle train/eval mode
            if trainable:
                module.train()
            else:
                module.eval()

    # RGB encoder
    def freeze_rgb_encoder(self):
        self._set_component_trainable("rgb_encoder", False)
    def unfreeze_rgb_encoder(self):
        self._set_component_trainable("rgb_encoder", True)

    # Depth encoder + fusion
    def freeze_depth_encoder(self):
        self._set_component_trainable("depth_encoder", False)
    def unfreeze_depth_encoder(self):
        self._set_component_trainable("depth_encoder", True)

    # Decoder head
    def freeze_decoder(self):
        self._set_component_trainable("decoder", False)
    def unfreeze_decoder(self):
        self._set_component_trainable("decoder", True)

    def forward(self, x: torch.Tensor):
        """
        x: (B,6,H,W)
        """
        B, C, H, W = x.shape
        tokens, (h, w) = self.encoder(x)          # (B, N, D)
        # per‐token classification
        logits = self.decoder(tokens)             # (B, N, out_classes)
        # reshape & upsample
        logits = logits.transpose(1, 2).view(B, -1, h, w)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

