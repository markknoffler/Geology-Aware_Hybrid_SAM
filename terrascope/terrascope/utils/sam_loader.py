import torch
import torch.nn as nn
from typing import Dict, Any

def load_sam_weights(model: nn.Module, sam_path: str):
    """
    Partial loading of SAM ViT-B weights into Terrascope.
    Maps:
      - image_encoder.patch_embed -> encoder.patch_embed
      - image_encoder.blocks.i -> encoder.layers[i].block_rgb
      - mask_decoder -> mask_decoder
    """
    print(f"Loading pre-trained SAM weights from {sam_path}...")
    try:
        sam_state = torch.load(sam_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load SAM weights: {e}")
        return

    # Handle both 'model' key and flat state dict
    state_dict = sam_state.get("model", sam_state)
    
    model_state = model.state_dict()
    new_state: Dict[str, Any] = {}

    # 1. Patch Embed
    for k, v in state_dict.items():
        if k.startswith("image_encoder.patch_embed."):
            new_key = k.replace("image_encoder.patch_embed.", "encoder.patch_embed.")
            if new_key in model_state:
                new_state[new_key] = v

    # 2. Transformer Blocks (RGB stream)
    for i in range(12):
        prefix = f"image_encoder.blocks.{i}."
        target_prefix = f"encoder.layers.{i}.block_rgb."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, target_prefix)
                if new_key in model_state:
                    new_state[new_key] = v

    # 3. Mask Decoder
    for k, v in state_dict.items():
        if k.startswith("mask_decoder."):
            if k in model_state:
                new_state[k] = v

    # 4. Handle size mismatches for Relative Position Embeddings
    for k in list(new_state.keys()):
        if "rel_pos_h" in k or "rel_pos_w" in k:
            v = new_state[k]
            m_v = model_state[k]
            if v.shape != m_v.shape:
                # Interpolate from checkpoint size to model size
                # Shape is (L, C), we interpolate along L
                v_reshaped = v.unsqueeze(0).permute(0, 2, 1) # (1, C, L)
                v_interp = torch.nn.functional.interpolate(
                    v_reshaped, size=m_v.shape[0], mode="linear", align_corners=False
                )
                new_state[k] = v_interp.permute(0, 2, 1).squeeze(0) # (L_new, C)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"Loaded {len(new_state)} tensors from SAM checkpoint.")
    print(f"Novel components (JointAttn, Fusion, GeoState, DEM-stream) initialized randomly.")
