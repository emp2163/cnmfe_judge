"""
cross_validation.py
===================
Leave-one-animal-out (LOAO) cross-validation framework for the dual-input
CNN ROI classifier.
 
Imports all data loading, model architecture, and training utilities directly
from cnn_classifier.py — no code duplication.
 
For each fold:
  - One animal is held out as the validation set
  - Remaining animals are used for training
  - A fresh model is trained from scratch
  - Performance metrics are saved per fold
 
A summary of metrics across all folds is printed and saved at the end.
 
Usage:
    python cross_validation.py
"""
 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch.utils.data import DataLoader
 
# ---------------------------------------------------------------------------
# Import everything from cnn_classifier — no duplication
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_classifier import (
    # config constants
    CROP_SIZE, PATCH_SIZE, TRACE_LENGTH, BATCH_SIZE,
    EPOCHS, LR, RANDOM_SEED, DS_FACTOR,
    OUTPUT_BASE,
    # model
    DualInputCNN,
    # data
    load_animal,
    # dataset class
    ROIDataset,
    # training utilities
    compute_class_weights,
    compute_metrics,
    train,
    # plotting
    plot_training_curves,
    plot_confusion_matrix,
)
 
 
# ---------------------------------------------------------------------------
# All animals — edit paths here to add animal 4 when ready
# ---------------------------------------------------------------------------
ALL_ANIMALS = [
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
    # --- add animal 4 here when ready ---
    # {
    #     "animal_id":    "XXXXXXAR",
    #     "cell_map_dir": r"path\to\cell_maps",
    #     "props_csv":    r"path\to\traces-props.csv",
    #     "traces_csv":   r"path\to\traces.csv",
    # },
]
 
 
# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------
 
def load_single_animal(cfg: dict) -> tuple:
    """Load one animal's data using cnn_classifier's load_animal()."""
    return load_animal(
        cfg['animal_id'],
        cfg['cell_map_dir'],
        cfg['props_csv'],
        cfg['traces_csv'],
        CROP_SIZE,
        PATCH_SIZE,
        DS_FACTOR,
        TRACE_LENGTH,
    )
 
 
def pool_animals(animal_data_list: list) -> tuple:
    """
    Pool data from multiple animals into a single dataset.
 
    Parameters
    ----------
    animal_data_list : list of (patches, traces, labels, names) tuples
 
    Returns
    -------
    patches, traces, labels, names — concatenated across animals
    """
    all_patches = np.concatenate([d[0] for d in animal_data_list], axis=0)
    all_traces  = np.concatenate([d[1] for d in animal_data_list], axis=0)
    all_labels  = np.concatenate([d[2] for d in animal_data_list], axis=0)
    all_names   = [name for d in animal_data_list for name in d[3]]
    return all_patches, all_traces, all_labels, all_names
 
 
def run_fold(fold_idx: int, val_animal_cfg: dict, train_animal_data: list,
             val_animal_data: tuple, device: str, out_dir: str) -> dict:
    """
    Run one fold of LOAO cross-validation.
 
    Parameters
    ----------
    fold_idx          : int   — fold number (0-indexed)
    val_animal_cfg    : dict  — config of the held-out animal
    train_animal_data : list  — list of (patches, traces, labels, names) for training animals
    val_animal_data   : tuple — (patches, traces, labels, names) for val animal
    device            : str
    out_dir           : str   — base output directory for this CV run
 
    Returns
    -------
    dict of final metrics for this fold
    """
    val_id = val_animal_cfg['animal_id']
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx+1} — Val animal: {val_id}")
    print(f"{'='*60}")
 
    # --- create fold output folder ---
    fold_dir = os.path.join(out_dir, f"fold_{fold_idx+1}_{val_id}")
    os.makedirs(fold_dir, exist_ok=True)
 
    # --- pool training data ---
    train_patches, train_traces, train_labels, _ = pool_animals(train_animal_data)
    val_patches,   val_traces,   val_labels,   _ = val_animal_data
 
    print(f"  Train: {len(train_patches)} cells — "
          f"{train_labels.sum()} accepted, {(train_labels==0).sum()} rejected")
    print(f"  Val:   {len(val_patches)} cells — "
          f"{val_labels.sum()} accepted, {(val_labels==0).sum()} rejected")
 
    # --- datasets ---
    train_ds = ROIDataset(train_patches, train_traces, train_labels, augment=True)
    val_ds   = ROIDataset(val_patches,   val_traces,   val_labels,   augment=False)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
 
    # --- fresh model for each fold ---
    model         = DualInputCNN(num_classes=2)
    class_weights = compute_class_weights(train_labels)
 
    # --- train ---
    model, history = train(
        model, train_loader, val_loader,
        EPOCHS, LR, class_weights, device
    )
 
    # --- save model ---
    model_path = os.path.join(fold_dir, f"model_fold{fold_idx+1}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved → {model_path}")
 
    # --- plots ---
    plot_training_curves(
        history,
        os.path.join(fold_dir, "training_curves.png")
    )
    plot_confusion_matrix(
        model, val_loader, device,
        os.path.join(fold_dir, "confusion_matrix.png")
    )
 
    # --- final metrics (from best epoch, already restored) ---
    best_f1        = max(history['val_f1'])
    best_epoch     = history['val_f1'].index(best_f1) + 1
    best_precision = history['val_precision'][best_epoch - 1]
    best_recall    = history['val_recall'][best_epoch - 1]
    best_acc       = history['val_acc'][best_epoch - 1]
 
    fold_metrics = {
        'fold':      fold_idx + 1,
        'val_animal': val_id,
        'n_train':   len(train_patches),
        'n_val':     len(val_patches),
        'best_epoch': best_epoch,
        'val_acc':   best_acc,
        'val_precision': best_precision,
        'val_recall':    best_recall,
        'val_f1':        best_f1,
    }
 
    print(f"\n  Fold {fold_idx+1} best results (epoch {best_epoch}):")
    print(f"    Accuracy:  {best_acc:.3f}")
    print(f"    Precision: {best_precision:.3f}")
    print(f"    Recall:    {best_recall:.3f}")
    print(f"    F1:        {best_f1:.3f}")
 
    return fold_metrics
 
 
def plot_cv_summary(all_metrics: list, save_path: str):
    """Bar chart of precision, recall, F1 per fold + mean line."""
    folds      = [f"Fold {m['fold']}\n({m['val_animal']})" for m in all_metrics]
    precisions = [m['val_precision'] for m in all_metrics]
    recalls    = [m['val_recall']    for m in all_metrics]
    f1s        = [m['val_f1']        for m in all_metrics]
 
    x   = np.arange(len(folds))
    w   = 0.25
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   precisions, w, label='Precision', color='steelblue')
    ax.bar(x,       recalls,    w, label='Recall',    color='seagreen')
    ax.bar(x + w,   f1s,        w, label='F1',        color='darkorange')
 
    # mean lines
    ax.axhline(np.mean(precisions), color='steelblue', linestyle='--',
               linewidth=1, alpha=0.7, label=f'Mean P={np.mean(precisions):.3f}')
    ax.axhline(np.mean(recalls),    color='seagreen',  linestyle='--',
               linewidth=1, alpha=0.7, label=f'Mean R={np.mean(recalls):.3f}')
    ax.axhline(np.mean(f1s),        color='darkorange', linestyle='--',
               linewidth=1, alpha=0.7, label=f'Mean F1={np.mean(f1s):.3f}')
 
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title('Leave-One-Animal-Out Cross-Validation Summary')
    ax.legend(loc='upper right', fontsize=8)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"CV summary plot saved → {save_path}")
 
 
def save_cv_summary_csv(all_metrics: list, save_path: str):
    """Save per-fold metrics and mean/std to CSV."""
    df = pd.DataFrame(all_metrics)
 
    # add mean and std rows
    numeric_cols = ['val_acc', 'val_precision', 'val_recall', 'val_f1']
    mean_row = {'fold': 'mean', 'val_animal': '—'}
    std_row  = {'fold': 'std',  'val_animal': '—'}
    for col in numeric_cols:
        mean_row[col] = df[col].mean()
        std_row[col]  = df[col].std()
 
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"CV summary CSV saved → {save_path}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Running {len(ALL_ANIMALS)}-fold leave-one-animal-out cross-validation")
 
    # --- create timestamped output folder ---
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_BASE, f"CV_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
 
    # --- preload all animal data once ---
    print("\nPreloading all animal data...")
    all_animal_data = []
    for cfg in ALL_ANIMALS:
        data = load_single_animal(cfg)
        all_animal_data.append(data)
 
    # --- run each fold ---
    all_metrics = []
    for fold_idx, val_cfg in enumerate(ALL_ANIMALS):
 
        # val = current animal, train = all others
        train_data = [d for i, d in enumerate(all_animal_data) if i != fold_idx]
        val_data   = all_animal_data[fold_idx]
 
        fold_metrics = run_fold(
            fold_idx   = fold_idx,
            val_animal_cfg   = val_cfg,
            train_animal_data = train_data,
            val_animal_data   = val_data,
            device     = device,
            out_dir    = out_dir,
        )
        all_metrics.append(fold_metrics)
 
    # --- summary ---
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Fold':<8} {'Val Animal':<14} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 50)
    for m in all_metrics:
        print(f"{m['fold']:<8} {m['val_animal']:<14} "
              f"{m['val_acc']:>6.3f} {m['val_precision']:>6.3f} "
              f"{m['val_recall']:>6.3f} {m['val_f1']:>6.3f}")
    print("-" * 50)
 
    accs  = [m['val_acc']       for m in all_metrics]
    precs = [m['val_precision'] for m in all_metrics]
    recs  = [m['val_recall']    for m in all_metrics]
    f1s   = [m['val_f1']        for m in all_metrics]
 
    print(f"{'Mean':<8} {'—':<14} "
          f"{np.mean(accs):>6.3f} {np.mean(precs):>6.3f} "
          f"{np.mean(recs):>6.3f} {np.mean(f1s):>6.3f}")
    print(f"{'Std':<8} {'—':<14} "
          f"{np.std(accs):>6.3f} {np.std(precs):>6.3f} "
          f"{np.std(recs):>6.3f} {np.std(f1s):>6.3f}")
 
    # --- save summary outputs ---
    plot_cv_summary(all_metrics,
                    os.path.join(out_dir, "cv_summary.png"))
    save_cv_summary_csv(all_metrics,
                        os.path.join(out_dir, "cv_summary.csv"))
 
    print(f"\nAll fold outputs saved to: {out_dir}")