import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTransposeDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            # reduce feature depth & add nonlinearity
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # upsample ×2
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # upsample ×2
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # upsample ×2
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # project to classes
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


class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, out_classes):
        super().__init__()
        self.requires_patch_divisible_input = True
        self.patch_size = 14
        self.depth_info = False
       
        self.rgb_encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        embed_dim = self.rgb_encoder.embed_dim
        self.out_classes = out_classes
        
        self.decoder = ConvTransposeDecoder(embed_dim, out_classes)
            
    def _set_component_trainable(self, name, trainable:bool):
            mp = {
                "rgb_encoder": [self.rgb_encoder],
                "decoder":      [self.decoder],
            }
            if name not in mp:
                raise ValueError(f"No such component: {name}")
            for module in mp[name]:
                for p in module.parameters():
                    p.requires_grad = trainable
                module.train() if trainable else module.eval()

        # convenience
    def freeze_rgb_encoder(self):     self._set_component_trainable("rgb_encoder", False)
    def unfreeze_rgb_encoder(self):   self._set_component_trainable("rgb_encoder", True)
    def freeze_decoder(self):         self._set_component_trainable("decoder", False)
    def unfreeze_decoder(self):       self._set_component_trainable("decoder", True)
 

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 3, "Expected 3 channels."
        Hn = (H // self.patch_size) * self.patch_size
        Wn = (W // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(Hn, Wn), mode='bilinear', align_corners=False)

        
        features = self.rgb_encoder.get_intermediate_layers(x, n=1)[0]  # Shape: (B, N, C)
        N, C = features.shape[1], features.shape[2]
        h, w = Hn//self.patch_size, Wn//self.patch_size
        feat = features.transpose(1,2).reshape(B, C, h, w)

        logits = self.decoder(feat, out_size = (H,W)) 
        
        return logits
