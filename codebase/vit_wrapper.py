import torch
import torch.nn as nn
from segment_anything.modeling.image_encoder import ImageEncoderViT  # Meta file[2]
from peg import PEG                                                 # ≤— fixed import
from cnn_adapter import CNNAdapter
import torch.nn.functional as F

class GImageEncoder(ImageEncoderViT):
    """
    SAM ImageEncoderViT + PEG + lightweight CNN fusion.
    """

    def __init__(self, *args, fuse_blocks=(4, 7, 10), **kwargs):
        # disable fixed absolute pos-emb
        kwargs["use_abs_pos"] = False
        super().__init__(*args, **kwargs)

        self.embed_dim = self.patch_embed.proj.out_channels
        self.fuse_at   = set(fuse_blocks)
        self.pegs      = nn.ModuleList([PEG(self.embed_dim) for _ in self.blocks])

        self.cnn_adapter = CNNAdapter(self.embed_dim)      # 1/16-scale branch
        self._fuse_proj  = nn.Conv2d(2 * self.embed_dim, self.embed_dim, 1)

    # --------------------------------------------------------
    def forward(self, img):                     # img B×3×H×W
        cnn_f = self.cnn_adapter(img)           # B×C×h×w  (h,w = H/16)
        x     = self.patch_embed(img)           # B×h×w×C
        H, W  = x.shape[1:3]

        for i, blk in enumerate(self.blocks):
            # 1. PEG
            seq = x.reshape(img.size(0), H * W, self.embed_dim)
            seq = self.pegs[i](seq, H, W)
            x   = seq.reshape(img.size(0), H, W, self.embed_dim)

            # 2. fuse CNN features at selected depths
            if i in self.fuse_at:
                x = self._fuse_grid(x, cnn_f)

            x = blk(x)                          # 3. vanilla ViT block

        return self.neck(x.permute(0, 3, 1, 2))  # B×256×h×w

    # --------------------------------------------------------
    def _fuse_grid(self, tokens_hwC, cnn_feat):
        B, H, W, C = tokens_hwC.shape  # ViT grid
        if cnn_feat.shape[2] != H or cnn_feat.shape[3] != W:
            cnn_feat = F.interpolate(cnn_feat, size=(H, W),
                                     mode="bilinear", align_corners=False)
        fused = torch.cat(
            [tokens_hwC.permute(0, 3, 1, 2),  # B×C×H×W
             cnn_feat], dim=1)  # B×2C×H×W
        fused = self._fuse_proj(fused)  # 1×1 back to C
        return fused.permute(0, 2, 3, 1)  # B×H×W×C

    # ---------- weight-transfer helper ----------
    # vit_wrapper.py  (inside class GImageEncoder)
    @classmethod
    def from_pretrained(cls, old_enc, **extra_cfg):
        embed_dim = old_enc.patch_embed.proj.out_channels
        hidden_dim = old_enc.blocks[0].mlp.lin1.out_features
        mlp_ratio = hidden_dim // embed_dim

        # find which blocks were global-attention in the source model
        global_ids = [i for i, b in enumerate(old_enc.blocks) if b.window_size == 0]

        genc = cls(
            img_size=old_enc.img_size,
            patch_size=old_enc.patch_embed.proj.kernel_size[0],
            in_chans=3,
            embed_dim=embed_dim,
            depth=len(old_enc.blocks),
            num_heads=old_enc.blocks[0].attn.num_heads,
            mlp_ratio=mlp_ratio,
            use_rel_pos=True,
            window_size=14,  # same as SAM
            global_attn_indexes=tuple(global_ids),
            **extra_cfg,
        )

        # remove only those bias tables whose shape still differs
        src = old_enc.state_dict()
        dst = genc.state_dict()
        for k, v in list(src.items()):
            if k in dst and v.shape != dst[k].shape:
                del src[k]

        genc.load_state_dict(src, strict=False)  # now succeeds
        return genc


