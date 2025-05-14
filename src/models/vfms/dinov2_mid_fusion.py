import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int, 
                 fusion_block: int = None, fuse_type: str = "concat"):
        """
        fusion_block: which transformer block index to fuse at.
                      If None, defaults to halfway through.
        fuse_type: "add"  (element‐wise sum) or "concat" (+ proj back).
        """
        super().__init__()


        self.depth_info = True
        # load RGB encoder
        self.rgb_encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        # duplicate for depth
        self.depth_encoder = copy.deepcopy(self.rgb_encoder)

        # patch size & shape flags
        self.patch_size = self.rgb_encoder.patch_embed.patch_size[0]
        self.requires_patch_divisible_input = True

        # determine fusion index
        total_blocks = len(self.rgb_encoder.blocks)
        self.fusion_block = fusion_block or (total_blocks // 2)
        assert 0 < self.fusion_block < total_blocks, "fusion_block out of range"

        # if concat fusion, project back to embed_dim
        self.fuse_type = fuse_type
        if fuse_type == "concat":
            ed = self.rgb_encoder.embed_dim
            self.fusion_proj = nn.Linear(2*ed, ed)

        # final linear decoder (pixel‐wise)
        self.decoder = nn.Linear(self.rgb_encoder.embed_dim, out_classes)

    def freeze_encoder(self):
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
        self.rgb_encoder.eval()
        for p in self.depth_encoder.parameters():
            p.requires_grad = False
        self.depth_encoder.eval()

    def unfreeze_encoder(self):
        for p in self.rgb_encoder.parameters():
            p.requires_grad = True
        self.rgb_encoder.train()
        for p in self.depth_encoder.parameters():
            p.requires_grad = True 
        self.depth_encoder.train()

    def _embed(self, x: torch.Tensor, encoder: nn.Module):
        # patch embedding + cls token + pos embed
        x = encoder.patch_embed(x)                      # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)                 # (B, N, C)
        cls_token = encoder.cls_token.expand(B, -1, -1) # (B,1,C)
        x = torch.cat((cls_token, x), dim=1)
        x = x + encoder.pos_embed
        return x

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # resize so H,W divisible by patch
        for t in (rgb, depth):
            assert t.shape[2] == rgb.shape[2] and t.shape[3] == rgb.shape[3], \
                "rgb/depth must have same spatial dims"

        new_h = (rgb.shape[2] // self.patch_size) * self.patch_size
        new_w = (rgb.shape[3] // self.patch_size) * self.patch_size
        rgb = F.interpolate(rgb, size=(new_h,new_w), mode='bilinear', align_corners=False)
        # depth: if single‐channel, unsqueeze; or replicate:
        if depth.shape[1] == 1:
            depth = depth.repeat(1,3,1,1)
        depth = F.interpolate(depth, size=(new_h,new_w), mode='bilinear', align_corners=False)

        # embed both streams
        x_rgb = self._embed(rgb, self.rgb_encoder)
        x_dep = self._embed(depth, self.depth_encoder)

        # feed through first half of transformer blocks
        for idx, block in enumerate(self.rgb_encoder.blocks):
            if idx == self.fusion_block:
                break
            x_rgb = block(x_rgb)
            x_dep = block(x_dep)

        # mid‐level fusion
        if self.fuse_type == "add":
            x = (x_rgb + x_dep) * 0.5
        else:  # concat
            x = torch.cat([x_rgb, x_dep], dim=-1)            # (B, N, 2C)
            x = self.fusion_proj(x)                          # → (B, N, C)

        # continue through remaining blocks
        for block in self.rgb_encoder.blocks[self.fusion_block:]:
            x = block(x)

        # normalization
        x = self.rgb_encoder.norm(x)  # (B, N, C)

        # drop cls token, go back to (B,C,H',W')
        x = x[:, 1:].transpose(1,2).view(
            rgb.size(0), self.rgb_encoder.embed_dim,
            new_h // self.patch_size, new_w // self.patch_size
        )

        # pixel‐wise linear & upsample
        logits = self.decoder(x.permute(0,2,3,1))
        logits = logits.permute(0,3,1,2)
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits
