# infer_gsam.py ──────────────────────────────────────────────────────────
import os, torch, torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ──── EDIT THE FOUR LINES BELOW ─────────────────────────────────────────
SAM_CKPT      = "/home/user5006/Documents/SAM/segment_anything/segment-anything/sam_vit_b_01ec64.pth"          # Meta’s original
MODEL_CKPT    = "/home/user5006/Documents/SAM/segment_anything/segment-anything/SAM_new_results/checkpoint_39.pt"     # your trained .pt
IN_DIR        = "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/landslide"                    # folder with RGBs
OUT_DIR       = "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/masks"          # empty or existing
# ─────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------
from builder import build_gsam_vit_b            # your custom builder

class GSAMWrapper(torch.nn.Module):
    """Keeps sub-module access and reproduces the training forward()"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, image_pe, dense_prompt):
        emb = self.model.image_encoder(img)        # ViT + PEG + CNNAdapter
        sparse_prompt = torch.zeros(emb.size(0), 0, 256, device=img.device)
        return self.model.mask_decoder(
            image_embeddings=emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False)                # logits, iou (iou unused)

# -----------------------------------------------------------------------
def load_model(device: torch.device) -> GSAMWrapper:
    base = build_gsam_vit_b(SAM_CKPT,
                            freeze_first_k=0,   # freezing irrelevant at inference
                            train_decoder=True).to(device)
    ckpt = torch.load(MODEL_CKPT, map_location=device)
    base.load_state_dict(ckpt["model_state_dict"], strict=False)
    return GSAMWrapper(base).eval()               # wrap + eval() mode

@torch.no_grad()
def infer_directory(model: GSAMWrapper,
                    in_dir:  str,
                    out_dir: str,
                    device:  torch.device):
    os.makedirs(out_dir, exist_ok=True)

    # Same preprocessing used during fine-tuning
    tfm = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    # Build positional encoding & dense prompt once
    pe_32 = model.model.prompt_encoder.pe_layer((32, 32)).to(device).unsqueeze(0)
    d_prompt = model.model.prompt_encoder.no_mask_embed.weight \
             .reshape(1, -1, 1, 1).expand(1, -1, 32, 32).to(device)

    for fname in tqdm(sorted(os.listdir(in_dir)), desc="segmenting"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue

        # 1. Load & preprocess image
        rgb = Image.open(os.path.join(in_dir, fname)).convert("RGB")
        inp = tfm(rgb).unsqueeze(0).to(device)           # shape 1×3×512×512

        # 2. Forward pass
        logits, _ = model(inp,
                  pe_32.expand(inp.size(0), -1, -1, -1),
                  d_prompt.expand(inp.size(0), -1, -1, -1))
        prob  = torch.sigmoid(logits)                    # 1×1×512×512
        mask  = (prob > 0.5).float()[0, 0]               # threshold @ 0.5

        # 3. Save binary mask (0 / 255 PNG)
        out_path = os.path.join(out_dir,
                                os.path.splitext(fname)[0] + "_mask.png")
        Image.fromarray((mask.cpu().numpy() * 255).astype("uint8")).save(out_path)

# -----------------------------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(DEVICE)
    infer_directory(model, IN_DIR, OUT_DIR, DEVICE)
    print(f"✓ Masks saved to {OUT_DIR}")
