import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MidFusionDINOv2Encoder(nn.Module):
    def __init__(self,
                 backbone: str,
                 fusion_block: int = None,
                 fuse_type: str = "concat",
                 depth_info: bool = True):
        super().__init__()
        # whether we're fusing depth at all
        self.depth_info = depth_info

        # load RGB stream and duplicate for depth
        self.rgb = torch.hub.load('facebookresearch/dinov2', backbone)
        self.depth = copy.deepcopy(self.rgb)

        # patch settings (for external cropping checks)
        self.patch_size = self.rgb.patch_embed.patch_size[0]
        self.requires_patch_divisible_input = True

        # which block to fuse in
        total = len(self.rgb.blocks)
        self.fusion_block = fusion_block or (total // 2)
        assert 0 < self.fusion_block < total

        # fusion mode
        self.fuse_type = fuse_type
        if fuse_type == "concat":
            ed = self.rgb.embed_dim
            self.fusion_proj = nn.Linear(2*ed, ed)

    def freeze(self):
        for stream in (self.rgb, self.depth):
            for p in stream.parameters():
                p.requires_grad = False
            stream.eval()

    def unfreeze(self):
        for stream in (self.rgb, self.depth):
            for p in stream.parameters():
                p.requires_grad = True
            stream.train()

    def _embed(self, x: torch.Tensor, enc: nn.Module):
        # patch‐embed + cls token + pos embed
        x = enc.patch_embed(x)                  # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)         # (B, N, C)
        cls = enc.cls_token.expand(B, -1, -1)   # (B, 1, C)
        x = torch.cat([cls, x], dim=1) + enc.pos_embed
        return x

    def forward(self, x: torch.Tensor):
        """
        x: [B, 6, H, W] if depth_info=True (rgb+hha),
           [B, 3, H, W] if depth_info=False (rgb only)
        """
        B, C, H, W = x.shape
        # split into rgb/depth
        if self.depth_info:
            assert C == 6, "Expected 6 channels (rgb+hha)"
            rgb, depth = x[:, :3], x[:, 3:]
        else:
            assert C == 3, "Expected 3 channels (rgb)"
            rgb, depth = x, None

        # crop to multiple of patch_size
        H0 = (H // self.patch_size) * self.patch_size
        W0 = (W // self.patch_size) * self.patch_size
        rgb = F.interpolate(rgb, size=(H0, W0), mode='bilinear', align_corners=False)
        if self.depth_info:
            depth = F.interpolate(depth, size=(H0, W0), mode='bilinear', align_corners=False)

        # embed streams
        x_rgb = self._embed(rgb, self.rgb)
        if self.depth_info:
            x_dep = self._embed(depth, self.depth)

        # feed through first half of blocks
        if self.depth_info:
            for i, blk in enumerate(self.rgb.blocks):
                if i == self.fusion_block:
                    break
                x_rgb = blk(x_rgb)
                x_dep = blk(x_dep)

            # fuse
            if self.fuse_type == "add":
                x = 0.5 * (x_rgb + x_dep)
            else:  # concat
                x = torch.cat([x_rgb, x_dep], dim=-1)
                x = self.fusion_proj(x)

            # remaining joint blocks
            for blk in self.rgb.blocks[self.fusion_block:]:
                x = blk(x)
        else:
            # pure‐rgb path
            x = x_rgb
            for blk in self.rgb.blocks:
                x = blk(x)

        # final norm
        x = self.rgb.norm(x)  # (B, N, C)

        # reshape back to feature map
        x = x[:, 1:].transpose(1, 2).reshape(
            B,
            self.rgb.embed_dim,
            H0 // self.patch_size,
            W0 // self.patch_size
        )
        return x


class DINOv2SegmentationModel(nn.Module):
    def __init__(self,
                 backbone: str,
                 out_classes: int,
                 fusion_block: int = None,
                 fuse_type: str = "concat",
                 depth_info: bool = True):
        super().__init__()
        # one unified encoder
        self.encoder = MidFusionDINOv2Encoder(
            backbone, fusion_block, fuse_type, depth_info
        )
        # decoder head
        self.decoder = nn.Linear(self.encoder.rgb.embed_dim, out_classes)

        # for external compatibility
        self.depth_info = depth_info
        self.patch_size = self.encoder.patch_size
        self.requires_patch_divisible_input = self.encoder.requires_patch_divisible_input

    def freeze_encoder(self):
        self.encoder.freeze()

    def unfreeze_encoder(self):
        self.encoder.unfreeze()

    def forward(self, inputs: torch.Tensor):
        """
        inputs: torch.Tensor of shape
          - [B, 6, H, W] if depth_info=True
          - [B, 3, H, W] if depth_info=False
        """
        feats = self.encoder(inputs)  # → (B, C, H', W')
        B, C, H, W = feats.shape

        # pixel‐wise linear head + upsample
        logits = self.decoder(feats.permute(0,2,3,1))  # (B, H', W', classes)
        logits = logits.permute(0,3,1,2)               # (B, classes, H', W')
        logits = F.interpolate(
            logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return logits

