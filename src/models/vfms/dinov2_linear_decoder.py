import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone, out_classes):
        super().__init__()
        self.requires_patch_divisible_input = True
        self.patch_size = 14
        self.depth_info = False
        self.encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        self.out_classes = out_classes
        self.decoder = nn.Linear(self.encoder.embed_dim, out_classes)

    def _set_component_trainable(self, component: str, trainable: bool):
        """
        component: one of {"rgb_encoder", "depth_encoder", "decoder"}
        trainable: True to unfreeze, False to freeze
        """
        if component == "rgb_encoder":
            modules = [self.encoder]
        elif component == "decoder":
            modules = [self.decoder]
        else:
            raise ValueError(f"Unknown component '{component}'")

        for module in modules:
            # toggle grad
            for p in module.parameters():
                p.requires_grad = trainable
            # toggle train/eval mode
            if trainable:
                module.train()
            else:
                module.eval()

    # RGB encoder
    def freeze_rgb_encoder(self):
        self._set_component_trainable("rgb_encoder", False)
    def unfreeze_rgb_encoder(self):
        self._set_component_trainable("rgb_encoder", True)

    # Decoder head
    def freeze_decoder(self):
        self._set_component_trainable("decoder", False)
    def unfreeze_decoder(self):
        self._set_component_trainable("decoder", True)


    def forward(self, x):
        new_height =  (x.shape[2] // self.patch_size) * self.patch_size 
        new_width =  (x.shape[3] // self.patch_size) * self.patch_size 
        x_resized = torch.nn.functional.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners = False)
        
        features = self.encoder.get_intermediate_layers(x_resized, n=1)[0]  # Shape: (B, N, C)
        B, N, C = features.shape
        H = (new_height // self.patch_size) 
        W = (new_width // self.patch_size)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        logits = self.decoder(features.permute(0, 2, 3, 1))  # (B, H, W, C)
        logits = logits.permute(0, 3, 1, 2)  # (B, C, H, W)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
