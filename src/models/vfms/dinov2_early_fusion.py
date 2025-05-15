import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone: str, out_classes: int ):
        super().__init__()
        self.requires_patch_divisible_input = True
        self.patch_size = 14
        self.depth_info = True

        # 1) Load pretrained DINOv2 (expects 3-channel input)
        self.encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        embed_dim = self.encoder.embed_dim

        # 2) Replace patch_embed conv to accept 6 channels (RGB+HHA)
        #    Original is Conv2d(3, embed_dim, patch_size, patch_size)
        orig_pe = self.encoder.patch_embed
        new_pe = nn.Conv2d(
            in_channels=6,
            out_channels=orig_pe.out_channels,
            kernel_size=orig_pe.kernel_size,
            stride=orig_pe.stride,
            padding=getattr(orig_pe, 'padding', 0)
        )
        # Copy RGB weights
        with torch.no_grad():
            new_pe.weight[:, :3, :, :] = orig_pe.weight
            # Initialize HHA weights as the mean over RGB kernels
            new_pe.weight[:, 3:, :, :] = orig_pe.weight.mean(dim=1, keepdim=True)
            new_pe.bias[:] = orig_pe.bias

        self.encoder.patch_embed = new_pe

        # 3) Simple linear decoder
        self.decoder = nn.Linear(embed_dim, out_classes)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.train()

    def forward(self, x: torch.Tensor):
        """
        x: Tensor[B, 6, H, W]  where channels = [R,G,B,H,H,A] (HHA_encoded depth)
        """
        # 1) enforce divisible by patch size
        H, W = x.shape[-2:]
        new_h = (H // self.patch_size) * self.patch_size
        new_w = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(new_h, new_w),
                          mode='bilinear', align_corners=False)

        # 2) run through DINOv2 encoder to get patch tokens
        #    get_intermediate_layers returns a list; use the first (last) layer
        tokens = self.encoder.get_intermediate_layers(x, n=1)[0]  # (B, N, C)
        B, N, C = tokens.shape

        # 3) reshape to feature map
        h_patches = new_h // self.patch_size
        w_patches = new_w // self.patch_size
        feats = tokens.permute(0, 2, 1).reshape(B, C, h_patches, w_patches)

        # 4) per-pixel logits
        #    decoder expects last-dim channels
        logits = self.decoder(feats.permute(0, 2, 3, 1))  # (B, Hp, Wp, out_classes)
        logits = logits.permute(0, 3, 1, 2)                # (B, out_classes, Hp, Wp)

        # 5) upsample back to input res
        return F.interpolate(logits, size=(H, W),
                             mode='bilinear', align_corners=False)