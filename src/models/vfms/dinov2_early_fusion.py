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

        # 2) Grab the original PatchEmbed and its Conv2d
        orig_pe = self.encoder.patch_embed             # this is a PatchEmbed object
        old_conv = orig_pe.proj                        # the Conv2d inside it

        # 3) Create a new Conv2d for 6 channels
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=getattr(old_conv, 'padding', 0),
            bias=(old_conv.bias is not None)
        )

        # 4) Initialize new_conv so that RGB weights are copied,
        #    and HHA channels start as the mean of RGB kernels
        with torch.no_grad():
            # copy the 3 RGB kernels
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # set the 3 HHA kernels to the mean over RGB
            new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias

        # 5) Replace the proj in the PatchEmbed, adjust in_chans for consistency
        orig_pe.proj = new_conv
        setattr(orig_pe, 'in_chans', 6)  # so any code that reads in_chans sees 6

        # 6) Simple linear decoder
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
        x: Tensor of shape (B, 6, H, W), where channels = [R, G, B, H, H, A]
        (the last three are your HHA-encoded depth).
        """
        # 1) make sure H, W divisible by patch_size
        B, C, H, W = x.shape
        new_h = (H // self.patch_size) * self.patch_size
        new_w = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(new_h, new_w),
                          mode='bilinear', align_corners=False)

        # 2) get the final patch tokens
        tokens = self.encoder.get_intermediate_layers(x, n=1)[0]  # (B, N, C_emb)
        B, N, C_emb = tokens.shape

        # 3) reshape back to spatial map
        h_p = new_h // self.patch_size
        w_p = new_w // self.patch_size
        feats = tokens.permute(0, 2, 1).reshape(B, C_emb, h_p, w_p)

        # 4) decode per-patch-class
        logits = self.decoder(feats.permute(0, 2, 3, 1))  # (B, h_p, w_p, out_classes)
        logits = logits.permute(0, 3, 1, 2)                # (B, out_classes, h_p, w_p)

        # 5) upsample to original H×W
        return F.interpolate(logits, size=(H, W),
                             mode='bilinear', align_corners=False)