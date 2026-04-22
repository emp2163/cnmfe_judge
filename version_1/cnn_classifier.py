"""
cnn_classifier.py
=================
Dual-input CNN-based ROI classifier for miniscope (1p) calcium imaging data.
 
Two input branches:
  - 2D CNN branch : spatial footprint (50x50 cell map crop)
  - 1D CNN branch : temporal dF/F trace (z-scored, downsampled to 10Hz)
 
Merged branches feed into a shared classification head.
 
Training animals : 184356AR, 152127AR
Validation animal: 15113AR
 
Outputs (saved to timestamped folder):
  - cnn_roi_classifier.pt
  - training_curves.png
 
Usage:
    python cnn_classifier.py
"""
 
import os
import re
import glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from datetime import datetime
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
TRAIN_ANIMALS = [
    {
        "animal_id":    "184356AR",
        "cell_map_dir": r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 2\184356",
        "props_csv":    r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 2\184356\184356AR-traces-props.csv",
        "traces_csv":   r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 2\184356\184356AR-traces.csv",
    },
    {
        "animal_id":    "152127AR",
        "cell_map_dir": r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 4\152127",
        "props_csv":    r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 4\152127\152127AR-traces-props.csv",
        "traces_csv":   r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 4\152127\152127AR-traces.csv",
    },
    {
        "animal_id":    "15113AR",
        "cell_map_dir": r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 1\151133",
        "props_csv":    r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 1\151133\15113AR-traces-props.csv",
        "traces_csv":   r"C:\Users\Salz_Lab\Documents\miniscope_mice\Preprocessed\Session 1\151133\15113AR-traces.csv",
    },
]
 
OUTPUT_BASE  = r"C:\Users\Salz_Lab\Documents\miniscope_mice\CNN_IStraining"
CROP_SIZE    = 50     # spatial crop window around centroid (pixels)
PATCH_SIZE   = 50     # CNN spatial input size
TRACE_LENGTH = 605    # timepoints after downsampling (1210 / 2)
BATCH_SIZE   = 16
EPOCHS       = 100
LR           = 1e-4
RANDOM_SEED  = 42
ORIG_FR      = 20     # original acquisition frame rate (Hz)
TARGET_FR    = 10     # target frame rate after downsampling (Hz)
DS_FACTOR    = ORIG_FR // TARGET_FR  # = 2
 
 
# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
 
class SpatialBranch(nn.Module):
    """2D CNN for spatial footprint (50x50x1)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=0)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.4)
 
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # same
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=0)  # valid
        self.bn4   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.4)
 
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(32 * 10 * 10, 128)
        self.bn5     = nn.BatchNorm1d(128)
        self.drop3   = nn.Dropout(0.4)
 
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # 50->48
        x = F.relu(self.bn2(self.conv2(x)))   # 48->46
        x = self.pool1(x)                      # 46->23
        x = self.drop1(x)
 
        x = F.relu(self.bn3(self.conv3(x)))   # 23->23
        x = F.relu(self.bn4(self.conv4(x)))   # 23->21
        x = self.pool2(x)                      # 21->10
        x = self.drop2(x)
 
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc(x)))
        return self.drop3(x)                   # (B, 128)
 
 
class TemporalBranch(nn.Module):
    """1D CNN for temporal dF/F trace (605 timepoints)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)           # 605 -> 151
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)           # 151 -> 37
        self.drop2 = nn.Dropout(0.3)
 
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.gap   = nn.AdaptiveAvgPool1d(1)   # -> (B, 64, 1)
 
        self.fc    = nn.Linear(64, 128)
        self.bn4   = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.4)
 
    def forward(self, x):
        # x: (B, 1, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
 
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
 
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)            # (B, 64)
 
        x = F.relu(self.bn4(self.fc(x)))
        return self.drop3(x)                   # (B, 128)
 
 
class DualInputCNN(nn.Module):
    """Merged spatial + temporal branches -> classification head."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.spatial  = SpatialBranch()
        self.temporal = TemporalBranch()
 
        # merged: 128 + 128 = 256
        self.fc1  = nn.Linear(256, 64)
        self.bn   = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.5)
        self.fc2  = nn.Linear(64, num_classes)
 
    def forward(self, patch, trace):
        s = self.spatial(patch)                # (B, 128)
        t = self.temporal(trace)               # (B, 128)
        x = torch.cat([s, t], dim=1)          # (B, 256)
        x = F.relu(self.bn(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)                     # raw logits
 
 
# ---------------------------------------------------------------------------
# Spatial data loading
# ---------------------------------------------------------------------------
 
def load_cell_map(tiff_path: str) -> np.ndarray:
    img = Image.open(tiff_path)
    return np.array(img, dtype=np.float32)
 
 
def crop_around_centroid(img: np.ndarray, cx: float, cy: float,
                          crop_size: int) -> np.ndarray:
    h, w   = img.shape
    half   = crop_size // 2
    cx, cy = int(round(cx)), int(round(cy))
 
    r0, r1 = cy - half, cy + half
    c0, c1 = cx - half, cx + half
 
    pad_top    = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left   = max(0, -c0)
    pad_right  = max(0, c1 - w)
 
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode='constant', constant_values=0)
        r0 += pad_top;  r1 += pad_top
        c0 += pad_left; c1 += pad_left
 
    return img[r0:r1, c0:c1]
 
 
def resize_patch(patch: np.ndarray, target: int) -> np.ndarray:
    h, w = patch.shape
    return zoom(patch, (target / h, target / w), order=1)
 
 
def normalize_patch(patch: np.ndarray) -> np.ndarray:
    mn, mx = patch.min(), patch.max()
    if mx - mn < 1e-8:
        return np.zeros_like(patch)
    return (patch - mn) / (mx - mn)
 
 
# ---------------------------------------------------------------------------
# Temporal data loading
# ---------------------------------------------------------------------------
 
def load_traces(traces_csv: str, ds_factor: int,
                target_length: int) -> dict:
    """
    Parse traces.csv, downsample temporally, z-score per cell.
 
    Returns dict: cell_name -> trace np.ndarray (target_length,)
    """
    df = pd.read_csv(traces_csv, header=0)
 
    # row 0 is the status row — drop it
    df = df.iloc[1:].reset_index(drop=True)
 
    # first column is time — drop it
    df = df.iloc[:, 1:]
 
    # strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
 
    df = df.astype(np.float32)
 
    traces = {}
    for col in df.columns:
        trace = df[col].values
 
        # temporal downsample
        trace = trace[::ds_factor]
 
        # pad or trim to target_length
        if len(trace) < target_length:
            trace = np.pad(trace, (0, target_length - len(trace)), mode='edge')
        else:
            trace = trace[:target_length]
 
        # z-score
        mu, sigma = trace.mean(), trace.std()
        trace = np.zeros_like(trace) if sigma < 1e-8 else (trace - mu) / sigma
 
        traces[col] = trace.astype(np.float32)
 
    return traces
 
 
# ---------------------------------------------------------------------------
# Load one animal
# ---------------------------------------------------------------------------
 
def load_animal(animal_id: str, cell_map_dir: str, props_csv: str,
                traces_csv: str, crop_size: int, patch_size: int,
                ds_factor: int, trace_length: int) -> tuple:
    props      = pd.read_csv(props_csv)
    props_dict = {row['Name']: row for _, row in props.iterrows()}
    trace_dict = load_traces(traces_csv, ds_factor, trace_length)
 
    tiff_files = sorted(glob.glob(os.path.join(cell_map_dir, '*-cells_C*.tiff')))
    if len(tiff_files) == 0:
        raise FileNotFoundError(f"No cell map TIFFs found in {cell_map_dir}")
 
    patches, traces_out, labels, names = [], [], [], []
 
    for fpath in tiff_files:
        match = re.search(r'(C\d+)\.tiff$', fpath)
        if match is None:
            continue
        cell_name = match.group(1)
 
        if cell_name not in props_dict:
            print(f"  {cell_name} not in props CSV, skipping")
            continue
        if cell_name not in trace_dict:
            print(f"  {cell_name} not in traces CSV, skipping")
            continue
 
        row    = props_dict[cell_name]
        status = row['Status'].strip().lower()
        label  = 1 if status == 'accepted' else 0
        cx, cy = row['CentroidX'], row['CentroidY']
 
        img   = load_cell_map(fpath)
        crop  = crop_around_centroid(img, cx, cy, crop_size)
        patch = resize_patch(crop, patch_size)
        patch = normalize_patch(patch)
 
        patches.append(patch)
        traces_out.append(trace_dict[cell_name])
        labels.append(label)
        names.append(cell_name)
 
    patches    = np.stack(patches)
    traces_out = np.stack(traces_out)
    labels     = np.array(labels)
 
    print(f"  {animal_id}: {len(patches)} cells — "
          f"{labels.sum()} accepted, {(labels==0).sum()} rejected")
    return patches, traces_out, labels, names
 
 
def build_dataset(animal_configs: list, crop_size: int, patch_size: int,
                  ds_factor: int, trace_length: int) -> tuple:
    all_patches, all_traces, all_labels, all_names = [], [], [], []
 
    for cfg in animal_configs:
        patches, traces, labels, names = load_animal(
            cfg['animal_id'], cfg['cell_map_dir'], cfg['props_csv'],
            cfg['traces_csv'], crop_size, patch_size, ds_factor, trace_length)
        all_patches.append(patches)
        all_traces.append(traces)
        all_labels.append(labels)
        all_names.extend(names)
 
    patches = np.concatenate(all_patches, axis=0)
    traces  = np.concatenate(all_traces,  axis=0)
    labels  = np.concatenate(all_labels,  axis=0)
 
    print(f"\nTotal pooled: {len(patches)} cells — "
          f"{labels.sum()} accepted, {(labels==0).sum()} rejected")
    return patches, traces, labels, all_names
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class ROIDataset(Dataset):
    def __init__(self, patches: np.ndarray, traces: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.patches = patches.astype(np.float32)
        self.traces  = traces.astype(np.float32)
        self.labels  = labels.astype(np.int64)
        self.augment = augment
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        patch = self.patches[idx].copy()
        trace = self.traces[idx].copy()
        label = self.labels[idx]
 
        if self.augment:
            # spatial augmentation
            if np.random.rand() > 0.5:
                patch = np.fliplr(patch).copy()
            if np.random.rand() > 0.5:
                patch = np.flipud(patch).copy()
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k).copy()
 
            # temporal augmentation: small random time shift
            shift = np.random.randint(-10, 10)
            trace = np.roll(trace, shift)
 
        patch = torch.tensor(patch).unsqueeze(0)  # (1, 50, 50)
        trace = torch.tensor(trace).unsqueeze(0)  # (1, 605)
        return patch, trace, label
 
 
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
 
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    n_total = len(labels)
    n_pos   = labels.sum()
    n_neg   = n_total - n_pos
    w_pos   = n_total / (2 * n_pos)
    w_neg   = n_total / (2 * n_neg)
    print(f"Class weights — accepted: {w_pos:.3f}, rejected: {w_neg:.3f}")
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)
 
 
def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray) -> dict:
    """
    Compute precision, recall, F1 for the positive class (accepted=1).
 
    precision = TP / (TP + FP)  — of cells called accepted, how many are correct
    recall    = TP / (TP + FN)  — of true accepted cells, how many did we find
    F1        = 2 * P * R / (P + R) — harmonic mean of precision and recall
    """
    TP = ((all_preds == 1) & (all_labels == 1)).sum()
    FP = ((all_preds == 1) & (all_labels == 0)).sum()
    FN = ((all_preds == 0) & (all_labels == 1)).sum()
 
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1}
 
 
def train(model, train_loader, val_loader, epochs, lr, class_weights, device):
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10, factor=0.5)
 
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss':   [], 'val_acc':   [], 'val_f1':   [],
               'val_precision': [], 'val_recall': [], 'lr': []}
 
    best_val_f1    = -1.0      # track best by F1 instead of loss
    best_weights   = None
    patience_count = 0
    early_stop_patience = 20
 
    for epoch in range(epochs):
        # --- train ---
        model.train()
        t_loss    = 0
        t_preds   = []
        t_labels  = []
 
        for patches, traces, lbls in train_loader:
            patches = patches.to(device)
            traces  = traces.to(device)
            lbls    = lbls.to(device)
 
            optimizer.zero_grad()
            logits = model(patches, traces)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
 
            t_loss   += loss.item() * len(lbls)
            t_preds.extend(logits.argmax(1).cpu().numpy())
            t_labels.extend(lbls.cpu().numpy())
 
        # --- validate ---
        model.eval()
        v_loss   = 0
        v_preds  = []
        v_labels = []
 
        with torch.no_grad():
            for patches, traces, lbls in val_loader:
                patches = patches.to(device)
                traces  = traces.to(device)
                lbls    = lbls.to(device)
 
                logits   = model(patches, traces)
                v_loss  += criterion(logits, lbls).item() * len(lbls)
                v_preds.extend(logits.argmax(1).cpu().numpy())
                v_labels.extend(lbls.cpu().numpy())
 
        # --- metrics ---
        t_preds  = np.array(t_preds)
        t_labels = np.array(t_labels)
        v_preds  = np.array(v_preds)
        v_labels = np.array(v_labels)
 
        t_acc  = (t_preds == t_labels).mean()
        v_acc  = (v_preds == v_labels).mean()
        t_l    = t_loss / len(t_labels)
        v_l    = v_loss / len(v_labels)
 
        t_metrics = compute_metrics(t_preds, t_labels)
        v_metrics = compute_metrics(v_preds, v_labels)
 
        history['train_loss'].append(t_l)
        history['train_acc'].append(t_acc)
        history['train_f1'].append(t_metrics['f1'])
        history['val_loss'].append(v_l)
        history['val_acc'].append(v_acc)
        history['val_f1'].append(v_metrics['f1'])
        history['val_precision'].append(v_metrics['precision'])
        history['val_recall'].append(v_metrics['recall'])
 
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
 
        scheduler.step(v_l)
 
        # early stopping on val F1 (more meaningful than val loss for imbalanced data)
        if v_metrics['f1'] > best_val_f1:
            best_val_f1  = v_metrics['f1']
            best_weights = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_count = 0
            star = " *"
        else:
            patience_count += 1
            star = ""
 
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"train loss {t_l:.4f} acc {t_acc:.3f} F1 {t_metrics['f1']:.3f} | "
              f"val loss {v_l:.4f} acc {v_acc:.3f} "
              f"P {v_metrics['precision']:.3f} R {v_metrics['recall']:.3f} "
              f"F1 {v_metrics['f1']:.3f} | lr {current_lr:.2e}{star}")
 
        if patience_count >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
 
    print(f"\nRestoring best model (val F1 {best_val_f1:.4f})")
    model.load_state_dict(best_weights)
    return model, history
 
 
def plot_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = range(1, len(history['train_loss']) + 1)
 
    # Loss
    axes[0,0].plot(epochs, history['train_loss'], label='train')
    axes[0,0].plot(epochs, history['val_loss'],   label='val')
    axes[0,0].set_title('Cross Entropy Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
 
    # Accuracy
    axes[0,1].plot(epochs, history['train_acc'], label='train')
    axes[0,1].plot(epochs, history['val_acc'],   label='val')
    axes[0,1].set_title('Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
 
    # F1
    axes[0,2].plot(epochs, history['train_f1'], label='train F1')
    axes[0,2].plot(epochs, history['val_f1'],   label='val F1')
    axes[0,2].set_title('F1 Score (accepted class)')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].legend()
 
    # Precision & Recall (val only)
    axes[1,0].plot(epochs, history['val_precision'], label='val precision', color='orange')
    axes[1,0].plot(epochs, history['val_recall'],    label='val recall',    color='green')
    axes[1,0].set_title('Val Precision & Recall (accepted class)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
 
    # Learning rate
    axes[1,1].plot(epochs, history['lr'], color='purple')
    axes[1,1].set_title('Learning Rate')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_yscale('log')
 
    # hide unused panel
    axes[1,2].axis('off')
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved → {save_path}")
 
 
def plot_confusion_matrix(model, val_loader, device, save_path: str):
    """Run model on full val set and plot confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
 
    with torch.no_grad():
        for patches, traces, lbls in val_loader:
            patches = patches.to(device)
            traces  = traces.to(device)
            logits  = model(patches, traces)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())
 
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
 
    # build 2x2 confusion matrix
    TP = ((all_preds == 1) & (all_labels == 1)).sum()
    FP = ((all_preds == 1) & (all_labels == 0)).sum()
    FN = ((all_preds == 0) & (all_labels == 1)).sum()
    TN = ((all_preds == 0) & (all_labels == 0)).sum()
 
    cm = np.array([[TN, FP],
                   [FN, TP]])
 
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
 
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred Rejected', 'Pred Accepted'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['True Rejected', 'True Accepted'])
 
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')
 
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
 
    ax.set_title(f'Confusion Matrix (Val)\n'
                 f'Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved → {save_path}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
 
    # --- create output folder ---
    run_id           = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir          = os.path.join(OUTPUT_BASE, run_id)
    os.makedirs(out_dir, exist_ok=True)
    SAVE_PATH        = os.path.join(out_dir, "cnn_roi_classifier.pt")
    CURVES_SAVE_PATH = os.path.join(out_dir, "training_curves.png")
    print(f"Output directory: {out_dir}")
 
    # --- load and pool all animals ---
    print("\nLoading all animals:")
    all_patches, all_traces, all_labels, all_names = build_dataset(
        TRAIN_ANIMALS, CROP_SIZE, PATCH_SIZE, DS_FACTOR, TRACE_LENGTH)
 
    # --- random 80/20 split ---
    n       = len(all_labels)
    idx     = np.random.permutation(n)
    n_val   = int(n * 0.2)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
 
    print(f"\nTrain: {len(tr_idx)} cells | Val: {len(val_idx)} cells")
 
    # --- datasets ---
    train_ds = ROIDataset(all_patches[tr_idx], all_traces[tr_idx],
                          all_labels[tr_idx],  augment=True)
    val_ds   = ROIDataset(all_patches[val_idx], all_traces[val_idx],
                          all_labels[val_idx],  augment=False)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
 
    # --- class weights from training split only ---
    class_weights = compute_class_weights(all_labels[tr_idx])
 
    # --- model ---
    model = DualInputCNN(num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
 
    # --- train ---
    model, history = train(model, train_loader, val_loader,
                           EPOCHS, LR, class_weights, device)
 
    # --- save ---
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved → {SAVE_PATH}")
 
    # --- plot ---
    plot_training_curves(history, CURVES_SAVE_PATH)
    plot_confusion_matrix(model, val_loader, device,
                          os.path.join(out_dir, "confusion_matrix.png"))