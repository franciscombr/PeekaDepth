import torch
import torch.nn as nn
import torch.nn.functional as F

class Mask2FormerDecoder(nn.Module):
    def __init__(self, 
                 in_dims,         # list of channel dims from backbone stages
                 pixel_dim=256,   # channels in pixel decoder
                 num_queries=100, # number of mask queries
                 transformer_dim=256,
                 num_heads=8,
                 num_layers=6,
                 num_classes=21):
        super().__init__()
        # 1. Pixel Decoder: fuse multi-scale features into one high-res map
        #    * simple lateral convs + upsampling
        self.laterals = nn.ModuleList([
            nn.Conv2d(d, pixel_dim, 1) for d in in_dims
        ])
        self.fpn_smooth = nn.ModuleList([
            nn.Conv2d(pixel_dim, pixel_dim, 3, padding=1) 
            for _ in in_dims[:-1]
        ])
        
        # 2. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim, nhead=num_heads, 
            dim_feedforward=transformer_dim*4
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.query_embed = nn.Embedding(num_queries, transformer_dim)
        
        # 3a. class head per query
        self.class_embed = nn.Linear(transformer_dim, num_classes)
        # 3b. mask embedding per query
        self.mask_embed = nn.Linear(transformer_dim, transformer_dim)

        # project pixel_dim → transformer_dim if needed
        if pixel_dim != transformer_dim:
            self.input_proj = nn.Conv2d(pixel_dim, transformer_dim, 1)
        else:
            self.input_proj = nn.Identity()

    def forward(self, backbone_feats):
        """
        backbone_feats: list of tensors from DINOv2 encoder, 
                        e.g. [f1, f2, f3] at resolutions H/4, H/8, H/16
        """
        # --- Pixel Decoder (FPN-style top-down) ---
        # lateral convs
        lat_feats = [l(f) for l, f in zip(self.laterals, backbone_feats)]
        # top-down
        for i in reversed(range(len(lat_feats)-1)):
            up = F.interpolate(lat_feats[i+1], size=lat_feats[i].shape[-2:], 
                               mode='nearest')
            lat_feats[i] = self.fpn_smooth[i](lat_feats[i] + up)
        pixel_feat = lat_feats[0]  # (B, pixel_dim, H, W)
        
        # project channels → transformer_dim
        pixel_embed = self.input_proj(pixel_feat)  # (B, D, H, W)
        B, D, H, W = pixel_embed.shape
        # flatten for attention: (H*W, B, D)
        pixel_flat = pixel_embed.flatten(2).permute(2, 0, 1)
        
        # --- Transformer Decoder ---
        # prepare query embeddings: (num_queries, B, D)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        # cross/self-attention
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer_decoder(tgt, pixel_flat, 
                                      memory_key_padding_mask=None,
                                      pos=None,
                                      query_pos=query_embed)
        # hs: (num_layers, num_queries, B, D) – take last layer
        hs = hs[-1]                          # (Q, B, D)
        hs = hs.permute(1, 0, 2)             # (B, Q, D)

        # --- Heads ---
        class_logits = self.class_embed(hs)  # (B, Q, num_classes)
        mask_embed = self.mask_embed(hs)     # (B, Q, D)

        # compute mask logits: 
        #   flatten pixel_embed to (B, D, H*W), then
        #   mask_embed @ pixel_flat_pixel for each query
        pixel_flat_for_mask = pixel_embed.flatten(2)  # (B, D, H*W)
        mask_logits = torch.einsum(
            "bqd, bdk -> bqk", 
            mask_embed, 
            pixel_flat_for_mask
        ).view(B, -1, H, W)  # (B, Q, H, W)

        return {"pred_masks": mask_logits,
                "pred_logits": class_logits}

class DINOv2Mask2Former(nn.Module):
    def __init__(self, backbone, out_classes, stages=(3,6,9)):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', backbone)
        self.patch_size = 14
        # pick a few intermediate layers to get multi-scale features
        self.stages = stages  
        embed_dim = self.encoder.embed_dim
        # Dimension of each chosen stage is embed_dim
        in_dims = [embed_dim]*len(stages)
        self.mask2former = Mask2FormerDecoder(
            in_dims=in_dims, 
            pixel_dim=256, 
            num_queries=100,
            transformer_dim=256,
            num_layers=6,
            num_heads=8,
            num_classes=out_classes
        )

    def forward(self, x):
        # crop to patch multiple
        H0 = (x.size(2)//self.patch_size)*self.patch_size
        W0 = (x.size(3)//self.patch_size)*self.patch_size
        x0 = F.interpolate(x, size=(H0, W0), 
                           mode='bilinear', align_corners=False)
        # get multiple layers
        feats = self.encoder.get_intermediate_layers(x0, 
                        n=max(self.stages)+1)  
        # select only the ones at your chosen depths
        backbone_feats = [feats[i] 
                          for i in self.stages]  
        # each is (B, N, C) → reshape to (B, C, Hf, Wf)
        pixel_feats = []
        Hf, Wf = H0//self.patch_size, W0//self.patch_size
        for f in backbone_feats:
            B, N, C = f.shape
            p = f.permute(0,2,1).reshape(B, C, Hf, Wf)
            pixel_feats.append(p)

        out = self.mask2former(pixel_feats)
        return out
