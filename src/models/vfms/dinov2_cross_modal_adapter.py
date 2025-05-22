import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthPatchEncoder(nn.Module):
    def __init__(self, backbone: str, patch_size:int = 14, num_heads:int = 8):
        super().__init__()
        # one conv = one “patch embedding”
        self.patch_size = patch_size

        self.rgb_backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        D = self.rgb_backbone.embed_dim
        self.depth_proj = nn.Conv2d(
            in_channels=3,
            out_channels=D,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x_depth):
        """
        x_depth: (B,3,H,W)  — 3‐channel depth map
        returns: (B, N, C)  — sequence of patch embeddings
        """
        # apply patch‐conv
        x = self.depth_proj(x_depth)         # (B, C, H/patch, W/patch)
        B, C, h, w = x.shape
        # flatten spatial → N = h*w
        x = x.flatten(2).transpose(1,2)  # (B, N, C)
        return x

class CrossAttentionAdapter(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, rgb_seq, depth_seq):
        # ─── ensure depth_seq is 3-D ─────────────────────────
        if depth_seq.dim() == 4:
            B, D, h, w = depth_seq.shape
            # flatten spatial dims into sequence
            depth_seq = depth_seq.flatten(2)          # (B, D, h*w)
            depth_seq = depth_seq.transpose(1, 2)     # (B, h*w, D)
        # now depth_seq is (B, N_depth, D)

        # ─── do cross‐attention ─────────────────────────────
        attn_out, _  = self.cross_attn(
            query=rgb_seq,   # (B, N_rgb, D)
            key=depth_seq,   # (B, N_depth, D)
            value=depth_seq  # (B, N_depth, D)
        )
        x = self.norm1(rgb_seq + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone:str, out_classes:int, patch_size:int=14, adapter_heads:int=8):
        super().__init__()
        self.patch_size = patch_size
        self.out_classes = out_classes

        # RGB DINOv2  gives embed_dim
        self.encoder = DepthPatchEncoder(backbone, patch_size, adapter_heads)
        C = self.encoder.rgb_backbone.embed_dim

        # cross-attention adapter
        self.adapter = CrossAttentionAdapter(C, num_heads=adapter_heads)

        # final per-patch classifier
        self.decoder = nn.Linear(C, out_classes)

    def _set_component_trainable(self, name, trainable:bool):
        mp = {
            "rgb_encoder": [self.encoder.rgb_backbone],
            "depth_encoder":[self.encoder.depth_proj, self.adapter],
            "decoder":      [self.decoder],
        }
        if name not in mp:
            raise ValueError(name)
        for m in mp[name]:
            for p in m.parameters():
                p.requires_grad = trainable
            m.train() if trainable else m.eval()

    # convenience
    def freeze_depth_encoder(self):   self._set_component_trainable("depth_encoder", False)
    def unfreeze_depth_encoder(self): self._set_component_trainable("depth_encoder", True)
    def freeze_rgb_encoder(self):         self._set_component_trainable("rgb_encoder", False)
    def unfreeze_rgb_encoder(self):       self._set_component_trainable("rgp_encoder", True)
    def freeze_decoder(self):         self._set_component_trainable("decoder", False)
    def unfreeze_decoder(self):       self._set_component_trainable("decoder", True)


    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == 6, "Expected 6 channels: RGB + 3 depth channels"
        Hn = (H // self.patch_size) * self.patch_size
        Wn = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(Hn, Wn), mode='bilinear', align_corners=False)

        rgb   = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        # 2) get sequences (B, N, C)
        rgb_seq   = self.encoder.rgb_backbone.get_intermediate_layers(rgb,   n=1)[0]
        depth_seq = self.encoder.depth_proj(depth)

        # 3) fuse
        fused_seq = self.adapter(rgb_seq, depth_seq)

        # 4) reshape to (B,C,h,w)
        N, C = fused_seq.shape[1], fused_seq.shape[2]
        h = Hn//self.patch_size
        w = Wn//self.patch_size
        feat = fused_seq.transpose(1,2).reshape(B, C, h, w)

        # 5) per-patch decode → per-pixel logits
        logits = self.decoder(feat.permute(0,2,3,1))  # (B,h,w,out)
        logits = logits.permute(0,3,1,2)               # (B,out,h,w)
        return F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
