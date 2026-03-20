# infer_gsam.py ──────────────────────────────────────────────────────────
import os, csv, torch, torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ──── EDIT THE FOUR LINES BELOW ─────────────────────────────────────────
SAM_CKPT      = "/home/user5006/Documents/SAM/segment_anything/segment-anything/sam_vit_b_01ec64.pth"
MODEL_CKPT    = "/home/user5006/Documents/SAM/segment_anything/segment-anything/SAM_new_results/checkpoint_39.pt"
IN_DIR        = "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/landslide"
OUT_DIR       = "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/confidence_masks"
# ─────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------
from builder import build_gsam_vit_b            # your custom builder

class GSAMWrapper(torch.nn.Module):
    """Keeps sub-module access and reproduces the training forward()"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img, image_pe, dense_prompt):
        emb = self.model.image_encoder(img)
        sparse_prompt = torch.zeros(emb.size(0), 0, 256, device=img.device)
        return self.model.mask_decoder(
            image_embeddings=emb,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False)              # logits, iou (confidence)

# -----------------------------------------------------------------------
def load_model(device: torch.device) -> GSAMWrapper:
    base = build_gsam_vit_b(SAM_CKPT,
                            freeze_first_k=0,
                            train_decoder=True).to(device)
    ckpt = torch.load(MODEL_CKPT, map_location=device)
    base.load_state_dict(ckpt["model_state_dict"], strict=False)
    return GSAMWrapper(base).eval()

@torch.no_grad()
def infer_directory(model: GSAMWrapper,
                    in_dir:  str,
                    out_dir: str,
                    device:  torch.device):
    os.makedirs(out_dir, exist_ok=True)

    # prepare CSV
    csv_path = os.path.join(out_dir, "confidence_scores.csv")
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["filename", "confidence"])      # header row

        # preprocessing identical to fine-tuning
        tfm = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

        # positional encoding & dense prompt (constant)
        pe_32 = model.model.prompt_encoder.pe_layer((32, 32)).to(device).unsqueeze(0)
        d_prompt = model.model.prompt_encoder.no_mask_embed.weight \
                 .reshape(1, -1, 1, 1).expand(1, -1, 32, 32).to(device)

        for fname in tqdm(sorted(os.listdir(in_dir)), desc="segmenting"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue

            # 1. load & preprocess
            rgb = Image.open(os.path.join(in_dir, fname)).convert("RGB")
            inp = tfm(rgb).unsqueeze(0).to(device)

            # 2. forward
            logits, iou = model(inp,
                                pe_32.expand(inp.size(0), -1, -1, -1),
                                d_prompt.expand(inp.size(0), -1, -1, -1))
            mask = (torch.sigmoid(logits) > 0.5).float()[0, 0]

            # 3. save mask
            out_path = os.path.join(out_dir,
                                    os.path.splitext(fname)[0] + "_mask.png")
            Image.fromarray((mask.cpu().numpy() * 255).astype("uint8")).save(out_path)

            # 4. write confidence to CSV
            writer.writerow([fname, f"{iou.item():.4f}"])

# -----------------------------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(DEVICE)
    infer_directory(model, IN_DIR, OUT_DIR, DEVICE)
    print(f"✓ Masks and confidence_scores.csv saved to {OUT_DIR}")
