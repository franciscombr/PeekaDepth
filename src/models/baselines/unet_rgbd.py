import torch
import torch.nn as nn
from torchvision import models

class DoubleConv(nn.Module):
    """
    Two sequential conv layers each followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetResNet(nn.Module):
    """
    U-Net with a pretrained ResNet encoder adapted for 4-channel RGB-D input.
    Outputs a segmentation map at the same spatial resolution as the input, using
    learnable transpose convolutions for upsampling.
    """
    def __init__(self, backbone='resnet34', pretrained=True, out_classes=40, freeze_encoder=False):
        super(UNetResNet, self).__init__()
        # Load pretrained ResNet
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Adapt first conv to take 4 channels (RGB + depth)
        orig_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            4,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = orig_conv.weight
            resnet.conv1.weight[:, 3:] = orig_conv.weight.mean(dim=1, keepdim=True)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False

        # Encoder layers
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Number of feature channels at each stage
        filters = [64, 64, 128, 256, 512]

        # Decoder path: upsample + double conv
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(filters[3] * 2, filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(filters[2] * 2, filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(filters[1] * 2, filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(filters[0] * 2, filters[0])

        # Additional learnable upsample to restore full resolution
        self.up0 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=2, stride=2)

        # Final segmentation head
        self.seg_head = nn.Conv2d(filters[0], out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward
        x0 = self.enc0(x)              # [B,64,H/2,W/2]
        x1 = self.enc1(self.pool0(x0)) # [B,64,H/4,W/4]
        x2 = self.enc2(x1)             # [B,128,H/8,W/8]
        x3 = self.enc3(x2)             # [B,256,H/16,W/16]
        x4 = self.enc4(x3)             # [B,512,H/32,W/32]

        # Decoder forward
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        # Final upsample and segmentation head
        d0 = self.up0(d1)               # [B,64,H,W]
        logits = self.seg_head(d0)      # [B, out_classes, H, W]
        return logits
