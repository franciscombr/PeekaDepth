import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTransposeDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
          # 1) project down & non-linearize
           nn.Conv2d(1024, 512, kernel_size=3, padding=1),
           nn.BatchNorm2d(512), nn.ReLU(inplace=True),

           # 2) ×2 → 68×90
           nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
           nn.BatchNorm2d(256), nn.ReLU(inplace=True),

           # 3) ×2 → 136×180
           nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
           nn.BatchNorm2d(128), nn.ReLU(inplace=True),

           # 4) ×2 → 272×360
           nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2),
           nn.BatchNorm2d(64), nn.ReLU(inplace=True),

           # 5) project to classes (still 272×360)
           nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]):
        """
        x: (B, in_channels, h, w)
        out_size: (H, W) – target spatial resolution
        """
        x = self.decode(x)
        # if decode doesn’t exactly hit (H, W), refine with bilinear
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size,
                              mode='bilinear', align_corners=False)
        return x

class DepthPatchEncoder(nn.Module):
    def __init__(self, embed_dim: int, patch_size:int = 14):
        super().__init__()
        self.patch_size = patch_size
        D = embed_dim
        self.depth_proj = nn.Sequential(
            nn.Conv2d(3, D, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(D), nn.GELU(),
            nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(D), nn.GELU(),
            nn.Conv2d(D, D, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x_depth):
        # x_depth: (B,3,H,W)
        x = self.depth_proj(x_depth)               # (B, C, H/patch, W/patch)
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1,2)            # (B, N, C)
        return x

class CrossAttentionAdapter(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.moddrop_p = 0

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.depth_norm = nn.LayerNorm(embed_dim)        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, rgb_seq, depth_seq):
        if depth_seq.dim() == 4:
            B, D, h, w = depth_seq.shape
            depth_seq = depth_seq.flatten(2).transpose(1, 2)

        if self.training and self.moddrop_p > 0:
            B, N, C = rgb_seq.shape
            keep_mask = (torch.rand(B, N, 1, device=rgb_seq.device) > self.moddrop_p).float()
            rgb_seq = rgb_seq * keep_mask

        rgb_seq = self.rgb_norm(rgb_seq)
        depth_seq = self.depth_norm(depth_seq)
        
        attn_out, _ = self.cross_attn(query=rgb_seq, key=depth_seq, value=depth_seq)
        x = rgb_seq + attn_out
        return x


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int,
                 patch_size: int = 14, adapter_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.out_classes = out_classes
        
        # 1) RGB encoder (DINOv2) 
        self.rgb_encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        embed_dim = self.rgb_encoder.embed_dim

        # 2) Depth encoder
        self.depth_encoder = DepthPatchEncoder(embed_dim, patch_size)

        # 3) Cross‐attention adapter
        self.adapter = CrossAttentionAdapter(embed_dim, num_heads=adapter_heads)

        self.decoder = ConvTransposeDecoder(embed_dim, out_classes)

    def _set_component_trainable(self, name, trainable:bool):
        mp = {
            "rgb_encoder": [self.rgb_encoder],
            "depth_encoder":[self.depth_encoder],
            "decoder":      [self.decoder],
            "adapter":      [self.adapter]
        }
        if name not in mp:
            raise ValueError(f"No such component: {name}")
        for module in mp[name]:
            for p in module.parameters():
                p.requires_grad = trainable
            module.train() if trainable else module.eval()

    # convenience
    def freeze_depth_encoder(self):   self._set_component_trainable("depth_encoder", False)
    def unfreeze_depth_encoder(self): self._set_component_trainable("depth_encoder", True)
    def freeze_rgb_encoder(self):     self._set_component_trainable("rgb_encoder", False)
    def unfreeze_rgb_encoder(self):   self._set_component_trainable("rgb_encoder", True)
    def freeze_decoder(self):         self._set_component_trainable("decoder", False)
    def unfreeze_decoder(self):       self._set_component_trainable("decoder", True)
    def freeze_adapter(self):         self._set_component_trainable("adapter", False)
    def unfreeze_adapter(self):       self._set_component_trainable("adapter", True)


    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == 6, "Expected 6 channels: RGB + 3 depth"
        Hn = (H // self.patch_size) * self.patch_size
        Wn = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(Hn, Wn), mode='bilinear', align_corners=False)

        rgb   = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # Extract embeddings
        rgb_seq   = self.rgb_encoder.get_intermediate_layers(rgb, n=1)[0]  # (B, N, C)
        depth_seq = self.depth_encoder(depth)                              # (B, N_depth, C)

        # Fuse
        fused_seq = self.adapter(rgb_seq, depth_seq)

        # Reshape back to spatial
        N, C = fused_seq.shape[1], fused_seq.shape[2]
        h, w = Hn//self.patch_size, Wn//self.patch_size
        feat = fused_seq.transpose(1,2).reshape(B, C, h, w)

        # Decode to per-pixel logits
        return self.decoder(feat, out_size=(H,W))



