# train_gsam.py ─────────────────────────────────────────────────────────
import os, csv, torch, tqdm, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset
from builder import build_gsam_vit_b
from my_dataset import SegDataset
from torch.nn.functional import binary_cross_entropy_with_logits as BCE
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ───────────── user-editable paths ────────────────────────────────────
CHECKPOINT_DIR        = "/home/user5006/Documents/SAM/segment_anything/segment-anything/SAM_new_results"
IMAGE_DIR             = "/home/user5006/Documents/SAM/segment_anything/segment-anything/color_padded/image"
MASK_DIR              = "/home/user5006/Documents/SAM/segment_anything/segment-anything/color_padded/masks"
# ----------------------------------------------------------------------

def dice(pred, target, eps: float = 1e-6):
    pred = torch.sigmoid(pred)
    num  = (pred * target).sum((2, 3))
    den  = pred.sum((2, 3)) + target.sum((2, 3))
    return 1 - ((2 * num + eps) / (den + eps)).mean()

class GSAMWrapper(torch.nn.Module):
    """keeps sub-module access when DataParallel is active"""
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
            multimask_output=False)

# ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "phase", "mean_loss", "precision", "recall", "accuracy", "auc"])

    # Multi-GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Using {device}  •  GPUs: {n_gpu}")
    #batch_size = 2 * n_gpu if n_gpu > 1 else 2
    batch_size = 2

    # 1 ▸ build model
    base_model = build_gsam_vit_b(
        "/home/user5006/Documents/SAM/segment_anything/segment-anything/sam_vit_b_01ec64.pth",
        freeze_first_k=8,
        train_decoder=True).to(device).train()

    model = GSAMWrapper(base_model)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    opt = torch.optim.AdamW(
        (p for p in base_model.parameters() if p.requires_grad),
        lr=1e-4, weight_decay=1e-2)

    # 2 ▸ dataset: single combined dataset
    full_ds = SegDataset(IMAGE_DIR, MASK_DIR)

    train_len = int(0.8 * len(full_ds))
    val_len   = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=1, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=1, pin_memory=torch.cuda.is_available())

    # 3 ▸ fixed pieces
    pe_32         = base_model.prompt_encoder.pe_layer((32, 32)).to(device)
    no_mask_embed = base_model.prompt_encoder.no_mask_embed.weight

    # Early stopping setup
    min_delta = 0.001
    best_precision = 0.0
    best_recall = 0.0
    patience = 5
    patience_counter = 0
    stop_training = False

    # 4 ▸ training loop
    for epoch in range(40):
        if stop_training:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        model.train()
        tr_loss = TP = FP = FN = TN = 0.0
        for img, mask in tqdm.tqdm(train_loader, desc=f"train {epoch:02d}", leave=False):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            B = img.size(0)
            img_pe  = pe_32.unsqueeze(0).expand(B, -1, -1, -1)
            d_prompt = no_mask_embed.reshape(1, -1, 1, 1).expand(B, -1, 32, 32)

            preds, _ = model(img, img_pe, d_prompt)
            preds    = F.interpolate(preds, (512, 512), mode="bilinear", align_corners=False)

            loss = BCE(preds, mask) + dice(preds, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()

            with torch.no_grad():
                prob  = torch.sigmoid(preds)
                predm = (prob > 0.5).float();  maskf = mask.float()
                TP += (predm * maskf).sum();            TN += ((1-predm)*(1-maskf)).sum()
                FP += (predm * (1 - maskf)).sum();      FN += ((1-predm)*maskf).sum()

        eps = 1e-6
        tr_P = TP / (TP + FP + eps);  tr_R = TP / (TP + FN + eps)
        tr_A = (TP + TN) / (TP + TN + FP + FN + eps)
        tr_L = tr_loss / len(train_loader)
        print(f"epoch {epoch:02d}  train loss={tr_L:.4f}  P={tr_P:.4f}  R={tr_R:.4f}  Acc={tr_A:.4f}")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, "train", f"{tr_L:.6f}", f"{tr_P:.6f}",
                                    f"{tr_R:.6f}", f"{tr_A:.6f}", ""])

        # 5 ▸ validation + checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval(); 
            val_loss = TP = FP = FN = TN = 0.0
            all_probs = []
            all_targets = []
            
            with torch.no_grad():
                for img, mask in tqdm.tqdm(val_loader, desc=f"val {epoch:02d}", leave=False):
                    img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                    B = img.size(0)
                    img_pe  = pe_32.unsqueeze(0).expand(B, -1, -1, -1)
                    d_prompt = no_mask_embed.reshape(1, -1, 1, 1).expand(B, -1, 32, 32)

                    preds, _ = model(img, img_pe, d_prompt)
                    preds    = F.interpolate(preds, (512, 512), mode="bilinear", align_corners=False)
                    loss_val = BCE(preds, mask) + dice(preds, mask)
                    val_loss += loss_val.item()

                    prob  = torch.sigmoid(preds)
                    all_probs.append(prob.cpu())
                    all_targets.append(mask.cpu())
                    
                    predm = (prob > 0.5).float();  maskf = mask.float()
                    TP += (predm * maskf).sum();            TN += ((1-predm)*(1-maskf)).sum()
                    FP += (predm * (1 - maskf)).sum();      FN += ((1-predm)*maskf).sum()

            # Calculate metrics
            v_P = TP / (TP + FP + eps);  v_R = TP / (TP + FN + eps)
            v_A = (TP + TN) / (TP + TN + FP + FN + eps)
            v_L = val_loss / len(val_loader)
            
            # Calculate AUC and plot ROC
            all_probs = torch.cat(all_probs).numpy().flatten()
            all_targets = torch.cat(all_targets).numpy().flatten()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            roc_path = os.path.join(CHECKPOINT_DIR, f"roc_epoch_{epoch:02d}.png")
            plt.savefig(roc_path)
            plt.close()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(all_targets, all_probs)
            pr_auc = auc(recall, precision)
            
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            pr_path = os.path.join(CHECKPOINT_DIR, f"pr_epoch_{epoch:02d}.png")
            plt.savefig(pr_path)
            plt.close()

            print(f"epoch {epoch:02d}  val   loss={v_L:.4f}  P={v_P:.4f}  R={v_R:.4f}  Acc={v_A:.4f}  AUC={roc_auc:.4f}")
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, "val", f"{v_L:.6f}", f"{v_P:.6f}",
                                        f"{v_R:.6f}", f"{v_A:.6f}", f"{roc_auc:.6f}"])

            # Save checkpoint
            ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_{epoch:02d}.pt")
            save_dict = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": opt.state_dict()
            }
            # Handle DataParallel wrapping
            if n_gpu > 1:
                save_dict["model_state_dict"] = model.module.model.state_dict()
            torch.save(save_dict, ckpt)
            print(f"✓ checkpoint saved to {ckpt}")
            
            # Early stopping check
            if v_P > best_precision + min_delta or v_R > best_recall + min_delta:
                best_precision = max(best_precision, v_P)
                best_recall = max(best_recall, v_R)
                patience_counter = 0
                print(f"Validation improved: P={v_P:.4f}, R={v_R:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in validation for {patience_counter}/5 epochs")
                
                if patience_counter >= patience:
                    stop_training = True
                    print("Early stopping criteria met. Stopping training...")
            
            model.train()

if __name__ == "__main__":
    main()