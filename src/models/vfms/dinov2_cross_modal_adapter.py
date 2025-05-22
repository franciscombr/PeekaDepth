import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthPatchEncoder(nn.Module):
    def __init__(self, patch_size:int, embed_dim:int):
        super().__init__()
        # one conv = one “patch embedding”
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x_depth):
        """
        x_depth: (B,1,H,W)  — single‐channel depth map
        returns: (B, N, C)  — sequence of patch embeddings
        """
        # apply patch‐conv
        x = self.proj(x_depth)         # (B, C, H/patch, W/patch)
        B, C, h, w = x.shape
        # flatten spatial → N = h*w
        x = x.flatten(2).transpose(1,2)  # (B, N, C)
        return x

class CrossAttentionAdapter(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, rgb_seq, depth_seq):
        attn_out, _ = self.cross_attn(rgb_seq, depth_seq, depth_seq)
        x = self.norm1(rgb_seq + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, backbone:str, out_classes:int, patch_size:int=14, adapter_heads:int=8):
        super().__init__()
        self.patch_size = patch_size
        self.out_classes = out_classes

        # RGB DINOv2  gives embed_dim
        self.rgb_encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        C = self.rgb_encoder.embed_dim

        # tiny depth patch encoder
        self.depth_encoder = DepthPatchEncoder(patch_size, C)

        # cross-attention adapter
        self.adapter = CrossAttentionAdapter(C, num_heads=adapter_heads)

        # final per-patch classifier
        self.decoder = nn.Linear(C, out_classes)

    def _set_component_trainable(self, name, trainable:bool):
        mp = {
            "rgb_encoder": [self.rgb_encoder],
            "depth_encoder":[self.depth_encoder, self.adapter],
            "decoder":      [self.decoder],
        }
        if name not in mp:
            raise ValueError(name)
        for m in mp[name]:
            for p in m.parameters():
                p.requires_grad = trainable
            m.train() if trainable else m.eval()

    # convenience
    def freeze_depth_encoder(self):   self._set_component_trainable("depth_encoder", False)
    def unfreeze_depth_encoder(self): self._set_component_trainable("depth_encoder", True)
    def freeze_rgb_encoder(self):         self._set_component_trainable("rgb_encoder", False)
    def unfreeze_rgb_encoder(self):       self._set_component_trainable("rgp_encoder", True)
    def freeze_decoder(self):         self._set_component_trainable("decoder", False)
    def unfreeze_decoder(self):       self._set_component_trainable("decoder", True)


    def forward(self, x_rgb, x_depth):
        B,_,H,W = x_rgb.shape
        # 1) resize both to multiples of patch_size
        Hn = (H//self.patch_size)*self.patch_size
        Wn = (W//self.patch_size)*self.patch_size
        rgb = F.interpolate(x_rgb,  size=(Hn,Wn), mode='bilinear', align_corners=False)
        depth = F.interpolate(x_depth, size=(Hn,Wn), mode='bilinear', align_corners=False)

        # 2) get sequences (B, N, C)
        rgb_seq   = self.rgb_encoder.get_intermediate_layers(rgb,   n=1)[0]
        depth_seq = self.depth_encoder(depth)

        # 3) fuse
        fused_seq = self.adapter(rgb_seq, depth_seq)

        # 4) reshape to (B,C,h,w)
        N, C = fused_seq.shape[1], fused_seq.shape[2]
        h = Hn//self.patch_size
        w = Wn//self.patch_size
        feat = fused_seq.transpose(1,2).reshape(B, C, h, w)

        # 5) per-patch decode → per-pixel logits
        logits = self.decoder(feat.permute(0,2,3,1))  # (B,h,w,out)
        logits = logits.permute(0,3,1,2)               # (B,out,h,w)
        return F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
