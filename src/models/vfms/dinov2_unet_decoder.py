import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_classes, num_ups=5, base_channels=None):
        super().__init__()
        # Determine base channel width if not given
        base_channels = base_channels or (in_channels // (2 ** num_ups))
        layers = []
        ch_in = in_channels
        for i in range(num_ups):
            ch_out = max(base_channels, ch_in // 2)
            layers.append(nn.ConvTranspose2d(
                ch_in, ch_out,
                kernel_size=2, stride=2,  # upsample ×2
                bias=False
            ))
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.up_blocks = nn.Sequential(*layers)
        # final classifier: ch_in → out_classes
        self.classifier = nn.Conv2d(ch_in, out_classes, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]  low-res patch features
        returns: [B, out_classes, H*2^num_ups, W*2^num_ups]
        """
        x = self.up_blocks(x)
        return self.classifier(x)


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, out_classes, decoder_cfg=None):
        super().__init__()
        self.requires_patch_divisible_input = True
        self.depth_info = False
        self.patch_size = 14
        self.encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        C = self.encoder.embed_dim
        # instantiate the UNetDecoder
        decoder_cfg = decoder_cfg or {}
        self.decoder = UNetDecoder(in_channels=C,
                                   out_classes=out_classes,
                                   **decoder_cfg)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.train()

    def forward(self, x):
        B, _, H0, W0 = x.shape
        # make dimensions divisible by patch_size
        H = (H0 // self.patch_size) * self.patch_size
        W = (W0 // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        # get patch tokens: shape (B, N, C)
        feat = self.encoder.get_intermediate_layers(x, n=1)[0]
        B, N, C = feat.shape
        h = H // self.patch_size
        w = W // self.patch_size
        feat = feat.permute(0, 2, 1).reshape(B, C, h, w)

        # decode to high‐res mask
        logits = self.decoder(feat)
        # finally, interpolate back to original input size
        return F.interpolate(logits, size=(H0, W0), mode='bilinear', align_corners=False)
