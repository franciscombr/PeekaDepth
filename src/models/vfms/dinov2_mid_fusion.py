import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MidFusionDINOv2Encoder(nn.Module):
    def __init__(self, backbone: str, fusion_block: int = None, fuse_type: str = "concat"):
        super().__init__()
        # load RGB encoder and duplicate for depth
        self.rgb = torch.hub.load('facebookresearch/dinov2', backbone)
        self.depth = copy.deepcopy(self.rgb)

        # patch/grid settings
        self.patch_size = self.rgb.patch_embed.patch_size[0]
        self.requires_patch_divisible_input = True

        # choose fusion layer
        total_blocks = len(self.rgb.blocks)
        self.fusion_block = fusion_block or (total_blocks // 2)
        assert 0 < self.fusion_block < total_blocks

        # fusion mode
        self.fuse_type = fuse_type
        if fuse_type == "concat":
            ed = self.rgb.embed_dim
            self.fusion_proj = nn.Linear(2*ed, ed)

    def freeze(self):
        for m in (self.rgb, self.depth):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def unfreeze(self):
        for m in (self.rgb, self.depth):
            for p in m.parameters():
                p.requires_grad = True
            m.train()

    def _embed(self, x, encoder):
        x = encoder.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)                 # (B, N, C)
        cls = encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + encoder.pos_embed
        return x

    def forward(self, rgb, depth):
        # align spatial dims
        H, W = rgb.shape[-2:]
        H0 = (H // self.patch_size) * self.patch_size
        W0 = (W // self.patch_size) * self.patch_size
        rgb = F.interpolate(rgb, size=(H0,W0), mode='bilinear', align_corners=False)

        # depth → 3ch
        if depth.shape[1] == 1:
            depth = depth.repeat(1,3,1,1)
        depth = F.interpolate(depth, size=(H0,W0), mode='bilinear', align_corners=False)

        # get token streams
        x_rgb = self._embed(rgb,   self.rgb)
        x_dep = self._embed(depth, self.depth)

        # first pass through each stream
        for i, blk in enumerate(self.rgb.blocks):
            if i == self.fusion_block:
                break
            x_rgb = blk(x_rgb)
            x_dep = blk(x_dep)

        # fuse tokens
        if self.fuse_type == "add":
            x = 0.5*(x_rgb + x_dep)
        else:
            x = torch.cat([x_rgb, x_dep], dim=-1)
            x = self.fusion_proj(x)

        # remaining joint blocks
        for blk in self.rgb.blocks[self.fusion_block:]:
            x = blk(x)
        x = self.rgb.norm(x)

        # reshape to feature‐map
        B, _, _ = x.shape
        x = x[:,1:].transpose(1,2).view(
            B, self.rgb.embed_dim,
            H0//self.patch_size, W0//self.patch_size
        )
        return x


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int,
                 fusion_block: int = None, fuse_type: str = "concat"):
        super().__init__()
        # one encoder object
        self.encoder = MidFusionDINOv2Encoder(backbone, fusion_block, fuse_type)
        # one decoder object: pixel‐wise linear head
        self.decoder = nn.Linear(self.encoder.rgb.embed_dim, out_classes)

    def freeze_encoder(self):
        self.encoder.freeze()

    def unfreeze_encoder(self):
        self.encoder.unfreeze()

    def forward(self, rgb, depth):
        # encoder → feature map (B, C, H', W')
        feats = self.encoder(rgb, depth)
        # decoder: apply linear per‐token then upsample
        B, C, H, W = feats.shape
        logits = self.decoder(feats.permute(0,2,3,1))  # (B, H, W, classes)
        logits = logits.permute(0,3,1,2)               # (B, classes, H, W)
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits
