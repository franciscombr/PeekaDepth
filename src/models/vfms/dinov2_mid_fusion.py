import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

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
        # conv → (B, C, Hʼ, Wʼ)
        x = enc.patch_embed.proj(x)
        B, C, Hf, Wf = x.shape

        # flatten to tokens → (B, N, C)
        x = x.flatten(2).transpose(1, 2)

        # prepend cls token → (B, N+1, C)
        cls = enc.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # ----- begin: interpolate pos_embed -----
        # enc.pos_embed: (1, orig_N+1, C)
        orig_embed = enc.pos_embed  # (1, P+1, C)
        cls_pos, grid_pos = orig_embed[:, :1], orig_embed[:, 1:]  # (1,1,C), (1,P,C)
        P = grid_pos.shape[1]
        orig_size = int(math.sqrt(P))
        # reshape → (1, C, orig_h, orig_w)
        grid_pos = grid_pos.view(1, orig_size, orig_size, C).permute(0, 3, 1, 2)
        # interpolate to (Hf, Wf)
        grid_pos = F.interpolate(
            grid_pos,
            size=(Hf, Wf),
            mode="bilinear",
            align_corners=False
        )
        # back to (1, Hf*Wf, C)
        grid_pos = grid_pos.permute(0, 2, 3, 1).view(1, Hf * Wf, C)
        # recombine
        new_pos_embed = torch.cat([cls_pos, grid_pos], dim=1)  # (1, Hf*Wf+1, C)
        # ----- end interpolation -----

        # now add them
        x = x + new_pos_embed
        return x, Hf, Wf

    def forward(self, x: torch.Tensor):
        """
        x: [B, 6, H, W] if depth_info=True (rgb+hha),
           [B, 3, H, W] if depth_info=False (rgb only)
        """
        B, C_in, H, W = x.shape
        # split into rgb/depth
        if self.depth_info:
            assert C_in == 6, "Expected 6 channels (rgb+hha)"
            rgb, depth = x[:, :3], x[:, 3:]
        else:
            assert C_in == 3, "Expected 3 channels (rgb)"
            rgb, depth = x, None

        # crop to multiple of patch_size
        H0 = (H // self.patch_size) * self.patch_size
        W0 = (W // self.patch_size) * self.patch_size
        rgb = F.interpolate(rgb, size=(H0, W0), mode='bilinear', align_corners=False)
        if self.depth_info:
            depth = F.interpolate(depth, size=(H0, W0), mode='bilinear', align_corners=False)

        # embed both streams (now gets H', W')
        x_rgb, Hf, Wf = self._embed(rgb,   self.rgb)
        if depth is not None:
            x_dep, _, _ = self._embed(depth, self.depth)

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
        x = x[:, 1:]                                   # drop cls token → (B, N, C)
        x = x.transpose(1, 2).view(                   # → (B, C, Hf, Wf)
            B, self.rgb.embed_dim, Hf, Wf
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

