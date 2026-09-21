"""
train_stage2.py
───────────────
Stage 2 fine-grained classifier: duck vs swan (vs other).
Input: 224×224 cropped ROI from Stage 1 bounding box.
Backbone: EfficientNet-B4 via timm (pretrained on ImageNet).
Loss: Supervised Contrastive Loss → pulls same-class embeddings together,
      pushes duck and swan embeddings apart in feature space.

Usage:
    python train_stage2.py --data ./stage2_dataset --run_name pond_s2_v1
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("timm not installed — run: pip install timm")

# ── Training config ────────────────────────────────────────────────────────────
CFG = dict(
    backbone    = 'efficientnet_b4',
    pretrained  = True,
    num_classes = 3,           # duck=0  swan=1  other=2
    img_size    = 224,
    batch_size  = 32,
    epochs      = 60,
    lr          = 2e-4,
    weight_decay= 1e-4,
    warmup_ep   = 5,
    # Contrastive loss temperature — lower = sharper separation in embedding space
    supcon_temp = 0.07,
    # Contrastive loss weight vs cross-entropy (anneal to 0 in later epochs)
    lambda_con  = 0.4,
    # Duck/swan misclassification is costlier than other/bird confusion
    class_weights = [1.0, 1.0, 0.5],   # duck, swan, other
)

CLASS_NAMES = ['duck', 'swan', 'other']


# ── Supervised Contrastive Loss ───────────────────────────────────────────────
class SupConLoss(nn.Module):
    """
    Khosla et al. (2020) — Supervised Contrastive Learning.
    Penalises the model when duck and swan embeddings are close in feature space.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # feats: (N, D) L2-normalised embeddings
        feats = F.normalize(feats, dim=1)
        sim   = torch.matmul(feats, feats.T) / self.temp   # (N, N)
        N     = feats.shape[0]
        mask  = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(feats.device)
        self_mask = torch.eye(N, device=feats.device)
        mask  = mask - self_mask         # exclude self-similarity
        exp   = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-8)
        loss  = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1).clamp(min=1))
        return loss.mean()


# ── Model with projection head ────────────────────────────────────────────────
class PondClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise RuntimeError("pip install timm required")
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0)   # remove head
        feat_dim = self.encoder.num_features
        # Projection head for SupCon (discarded at inference)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, 128)
        )
        # Classification head
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feats=False):
        feats  = self.encoder(x)
        logits = self.head(feats)
        if return_feats:
            proj = self.proj(feats)
            return logits, proj
        return logits


# ── Data loaders ──────────────────────────────────────────────────────────────
def make_loaders(data_root: str, img_size: int, batch: int):
    # Training augmentation
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    root = Path(data_root)
    train_ds = datasets.ImageFolder(root/'train', transform=train_tfm)
    val_ds   = datasets.ImageFolder(root/'val',   transform=val_tfm)
    test_ds  = datasets.ImageFolder(root/'test',  transform=val_tfm)

    # Weighted sampler to counter duck/swan vs other imbalance
    counts  = [0] * len(train_ds.classes)
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    weights = [1.0/counts[lbl] for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                          num_workers=4, pin_memory=True)

    print(f"  Classes: {train_ds.classes}")
    print(f"  Train {len(train_ds)}  Val {len(val_ds)}  Test {len(test_ds)}")
    return train_dl, val_dl, test_dl, train_ds.classes


# ── Evaluation with confusion matrix ─────────────────────────────────────────
def evaluate(model, loader, device, class_names):
    model.eval()
    n = len(class_names)
    conf = [[0]*n for _ in range(n)]   # conf[true][pred]
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                conf[t][p] += 1
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total

    # Print confusion matrix
    width = max(len(c) for c in class_names) + 2
    header = ' ' * (width+2) + '  '.join(f"{c:>{width}}" for c in class_names)
    print(header)
    for i, row in enumerate(conf):
        cells = []
        for j, v in enumerate(row):
            mark = ' ❌' if (i != j and class_names[i] in ('duck','swan')
                             and class_names[j] in ('duck','swan')) else ''
            cells.append(f"{v:>{width}}{mark}")
        print(f"  {class_names[i]:<{width}}  " + '  '.join(cells))

    # Key metric: duck↔swan swap rate
    duck_i = class_names.index('duck') if 'duck' in class_names else -1
    swan_i = class_names.index('swan') if 'swan' in class_names else -1
    if duck_i >= 0 and swan_i >= 0:
        d_total  = sum(conf[duck_i])
        s_total  = sum(conf[swan_i])
        d2s = conf[duck_i][swan_i] / max(d_total, 1)
        s2d = conf[swan_i][duck_i] / max(s_total, 1)
        print(f"\n  duck→swan swap : {d2s*100:.1f}%   (target < 5%)")
        print(f"  swan→duck swap : {s2d*100:.1f}%   (target < 5%)")

    return acc


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    train_dl, val_dl, test_dl, class_names = make_loaders(
        args.data, CFG['img_size'], CFG['batch_size'])

    model = PondClassifier(CFG['backbone'], CFG['num_classes'],
                            CFG['pretrained']).to(device)

    w = torch.tensor(CFG['class_weights'], dtype=torch.float32).to(device)
    ce_loss  = nn.CrossEntropyLoss(weight=w)
    con_loss = SupConLoss(temperature=CFG['supcon_temp'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'])

    best_val_acc = 0.0
    out_dir = Path('runs/pond_stage2') / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, CFG['epochs'] + 1):
        model.train()
        total_loss = 0.0
        # Anneal contrastive weight to 0 in final 20 epochs
        lam = CFG['lambda_con'] * max(0, 1 - (ep - CFG['epochs'] + 20) / 20)

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if lam > 0:
                logits, proj = model(imgs, return_feats=True)
                loss = ce_loss(logits, labels) + lam * con_loss(proj, labels)
            else:
                logits = model(imgs)
                loss = ce_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate every 5 epochs
        if ep % 5 == 0 or ep == CFG['epochs']:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += (model(imgs).argmax(1) == labels).sum().item()
                    total   += labels.size(0)
            val_acc = correct / total
            improved = '✅' if val_acc > best_val_acc else ''
            print(f"  Ep {ep:03d} | loss {total_loss/len(train_dl):.4f} "
                  f"| val_acc {val_acc:.4f} | λ_con={lam:.3f} {improved}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), out_dir / 'best.pt')

    # ── Final test evaluation ──────────────────────────────────────────────────
    print(f"\n── Test set evaluation (best checkpoint) ──────────────────────")
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
    test_acc = evaluate(model, test_dl, device, class_names)
    print(f"\n  Test accuracy: {test_acc:.4f}")
    print(f"  Best weights → {out_dir / 'best.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',     default='./stage2_dataset')
    ap.add_argument('--run_name', default='pond_s2_v1')
    ap.add_argument('--device',   default='0')
    args = ap.parse_args()
    train(args)

if __name__ == '__main__':
    main()
