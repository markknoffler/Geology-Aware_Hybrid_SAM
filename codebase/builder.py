# gsam/builder.py
from segment_anything.build_sam import build_sam_vit_b          # Meta’s factory
from vit_wrapper import GImageEncoder                            # PEG-+-CNN wrapper


def build_gsam_vit_b(
    ckpt_path: str,
    freeze_first_k: int = 4,     # 0 ⇒ train all 12 ViT blocks, 4 ⇒ last 8, 8 ⇒ last 4
    train_decoder: bool = True,  # False ⇒ keep the whole mask-decoder frozen
) -> "Sam":
    """
    Return a SAM-B model whose image encoder is replaced by the PEG + CNN
    wrapper.  `requires_grad` flags are set so that:

      • prompt-encoder is always frozen
      • the first `freeze_first_k` ViT blocks are frozen
      • every PEG kernel, every CNN-Adapter weight, the fusion 1×1 conv
        and the remaining ViT blocks stay trainable
      • the mask-decoder is trainable unless `train_decoder=False`

    Parameters
    ----------
    ckpt_path       : Path to Meta’s original `sam_vit_b_01ec64.pth`.
    freeze_first_k  : How many early transformer blocks to keep frozen.
    train_decoder   : Whether to fine-tune the mask-decoder.
    """
    # 1  load vanilla SAM-B weights
    sam = build_sam_vit_b(checkpoint=ckpt_path)
    sam.train()                                       # undo .eval() in factory

    # 2  swap encoder for the GSAM wrapper (PEG + CNN fusion)
    sam.image_encoder = GImageEncoder.from_pretrained(sam.image_encoder)

    # 3  freeze prompt-encoder
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # 4  selectively freeze ViT blocks
    for i, blk in enumerate(sam.image_encoder.blocks):
        for p in blk.parameters():
            p.requires_grad = i >= freeze_first_k      # keep early blocks frozen

    # 5  optionally freeze the entire mask-decoder
    if not train_decoder:
        for p in sam.mask_decoder.parameters():
            p.requires_grad = False

    # PEGs, CNN-Adapter, fusion 1×1 and neck remain trainable by default
    return sam
