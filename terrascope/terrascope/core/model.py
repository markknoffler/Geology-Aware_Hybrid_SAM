import torch
import torch.nn as nn

from terrascope.core.encoder import TerrascopeEncoder
from terrascope.core.mask_decoder import MaskDecoder
from terrascope.core.position_encoding import PositionEmbeddingRandom
from terrascope.core.transformer import TwoWayTransformer


class TerrascopePromptBundle(nn.Module):
    """Minimal dense PE + no-mask token (SAM prompt encoder subset, trainable from scratch)."""

    def __init__(self, embed_dim: int = 256, image_embedding_size: tuple[int, int] = (64, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.landslide_token = nn.Embedding(1, embed_dim)

    def dense_pe(self, spatial_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        return self.pe_layer(spatial_size).unsqueeze(0).to(device)


class Terrascope(nn.Module):
    """
    Terrascope: dual-stream landslide segmentation. Encoder is native multi-stream;
    mask branch reuses SAM MaskDecoder + TwoWayTransformer topology (random init).
    """

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 16,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_global_attn_indexes: tuple[int, ...] = (2, 5, 8, 11),
        prompt_embed_dim: int = 256,
        dem_in_chans: int = 1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.prompt_embed_dim = prompt_embed_dim
        self.encoder = TerrascopeEncoder(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            out_chans=prompt_embed_dim,
            global_attn_indexes=encoder_global_attn_indexes,
            dem_in_chans=dem_in_chans,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        emb_hw = image_size // patch_size
        self.prompts = TerrascopePromptBundle(
            embed_dim=prompt_embed_dim, image_embedding_size=(emb_hw, emb_hw)
        )
        self.aux_head_rgb = nn.Conv2d(encoder_embed_dim, 1, kernel_size=1)
        self.aux_head_dem = nn.Conv2d(encoder_embed_dim, 1, kernel_size=1)

    def forward(
        self,
        rgb: torch.Tensor,
        dem: torch.Tensor,
        image_pe: torch.Tensor,
        dense_prompt: torch.Tensor,
        multimask_output: bool = False,
        return_aux: bool = False,
    ):
        enc, mid = self.encoder(rgb, dem, return_mid=return_aux)
        b = rgb.size(0)
        # Use a learnable landslide token as a global query prompt
        sparse = self.prompts.landslide_token.weight.unsqueeze(0).expand(b, -1, -1)
        masks, iou = self.mask_decoder(
            image_embeddings=enc,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=multimask_output,
        )
        aux = None
        if return_aux and mid is not None:
            rgb_m, dem_m = mid
            aux = (
                self.aux_head_rgb(rgb_m.permute(0, 3, 1, 2)),
                self.aux_head_dem(dem_m.permute(0, 3, 1, 2)),
            )
        return masks, iou, aux


def build_terrascope_b(
    *,
    image_size: int = 1024,
    patch_size: int = 16,
    dem_in_chans: int = 1,
) -> Terrascope:
    """ViT-B-scale Terrascope (SAM-B patch geometry); random init. `image_size` must match training crop."""
    return Terrascope(
        image_size=image_size,
        patch_size=patch_size,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        prompt_embed_dim=256,
        dem_in_chans=dem_in_chans,
    )
