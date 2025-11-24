import os
import time
import random
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import amp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve
)
from scipy.stats import pearsonr

# ---------------------------
# 0. Configuration
# ---------------------------
DATA_PATH = "Dataset_AVDOSVR_postprocessed.csv"
MODEL_NAME = "CNN"
RESULTS_ROOT = "results"
MODELS_ROOT = "models"
SEED = 42
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 10 
BATCH_INFER_WARMUP = 64
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False
PRINT_EPOCH_SUMMARY = True

# Derived dirs
RESULTS_DIR = os.path.join(RESULTS_ROOT, MODEL_NAME)
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_ROOT, MODEL_NAME), exist_ok=True)

# ---------------------------
# 1. Utilities
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def concordance_ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return (2 * cov) / denom if denom != 0 else 0.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def savefig(path):
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

# ---------------------------
# 2. Model - CNNModel
# ---------------------------
class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_channels=64, kernel_size=3, dropout=0.3, output_size=2):
        super().__init__()
        # Conv1d expects (batch, in_channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout)
        
        self.flat_size = None # Determined dynamically
        self.fc1 = nn.Linear(1024, 128) # Placeholder size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, output_size)
        self.swish = nn.SiLU()

        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None: nn.init.zeros_(conv.bias)
        for layer in (self.fc1, self.fc2, self.fc3, self.fc_out):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x: (batch, num_features, 1) -> permute -> (batch, 1, num_features)
        x = x.permute(0, 2, 1) 

        out = self.pool1(self.swish(self.bn1(self.conv1(x))))
        out = self.pool2(self.swish(self.bn2(self.conv2(out))))
        
        if self.flat_size is None:
            self.flat_size = out.shape[1] * out.shape[2]
            self.fc1 = nn.Linear(self.flat_size, 128).to(out.device)
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)

        out = out.flatten(start_dim=1)
        out = self.dropout(out)
        out = self.swish(self.fc1(out))
        out = self.swish(self.fc2(out))
        out = self.swish(self.fc3(out))
        out = self.fc_out(out)
        return out

# ---------------------------
# 3. Load and preprocess data
# ---------------------------
print("[INFO] Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

for c in ['Participant', 'OriginalParticipantID', 'VideoId', 'Stage']:
    if c in df.columns:
        df = df.drop(columns=c)

df = df.dropna().reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ['Valence','Arousal']]
print(f"[INFO] Samples: {len(df)}, Columns found: {len(df.columns)} -> Features: {len(feature_cols)}")

X_raw = df[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)
if X_raw.shape[1] != len(feature_cols):
    print("[WARN] Some columns were non-numeric and were dropped during preprocessing.")
X = X_raw
Y = df[['Valence','Arousal']].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rng = np.random.RandomState(0)
weights = rng.rand(X_scaled.shape[1]).astype(np.float32)
weighted_feature = (X_scaled * weights).sum(axis=1, keepdims=True).astype(np.float32)
X_aug = np.hstack((X_scaled, weighted_feature))

# reshape to (N, seq_len=num_features, input_size=1)
X_seq = X_aug.reshape(X_aug.shape[0], X_aug.shape[1], 1).astype(np.float32)

val_median_global = np.median(Y[:,0])
stratify_bins = (Y[:,0] >= val_median_global).astype(int)

# ---------------------------
# 4. Cross validation setup
# ---------------------------
set_seed(SEED)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

per_fold_metrics = []
fold_idx = 0
start_time_all = time.time()

for train_idx, test_idx in skf.split(X_seq, stratify_bins):
    fold_idx += 1
    print(f"\n[INFO] Starting fold {fold_idx}/{N_SPLITS}")

    X_train = X_seq[train_idx]
    X_test = X_seq[test_idx]
    y_train = Y[train_idx]
    y_test = Y[test_idx]

    val_thr = np.median(y_train[:,0])
    aro_thr = np.median(y_train[:,1])

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model instantiation (REPLACED)
    model = CNNModel(input_size=1, hidden_channels=64, kernel_size=3, dropout=0.3, output_size=2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    amp_scaler = amp.GradScaler(device='cuda') if USE_MIXED_PRECISION and torch.cuda.is_available() else None

    param_count = count_parameters(model)
    print(f"[INFO] Params: {param_count}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    train_loss_hist = []
    val_loss_hist = []
    train_mae_hist = []
    val_mae_hist = []
    epoch_durations = []

    fold_train_start = time.time()

    for epoch in range(1, EPOCHS+1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        n_batches = 0

        loop = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}/{EPOCHS} (train)", leave=False)
        for xb, yb in loop:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()

            if amp_scaler is not None:
                with amp.autocast(device_type='cuda'):
                    out = model(xb)
                    loss = criterion(out, yb)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_mae += torch.mean(torch.abs(out - yb)).item()
            n_batches += 1
            loop.set_postfix(train_loss = running_loss / n_batches, train_mae = running_mae / n_batches)

        avg_train_loss = running_loss / max(1, n_batches)
        avg_train_mae = running_mae / max(1, n_batches)
        train_loss_hist.append(avg_train_loss)
        train_mae_hist.append(avg_train_mae)

        # validation
        model.eval()
        v_loss = 0.0
        v_mae = 0.0
        v_batches = 0
        with torch.no_grad():
            loopv = tqdm(val_loader, desc=f"Fold {fold_idx} Epoch {epoch}/{EPOCHS} (val)", leave=False)
            for xb, yb in loopv:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                if amp_scaler is not None:
                    with amp.autocast(device_type='cuda'):
                        out = model(xb)
                        loss = criterion(out, yb)
                else:
                    out = model(xb)
                    loss = criterion(out, yb)
                v_loss += loss.item()
                v_mae += torch.mean(torch.abs(out - yb)).item()
                v_batches += 1
                loopv.set_postfix(val_loss = v_loss / v_batches if v_batches else 0, val_mae = v_mae / v_batches if v_batches else 0)

        avg_val_loss = v_loss / max(1, v_batches)
        avg_val_mae = v_mae / max(1, v_batches)
        val_loss_hist.append(avg_val_loss)
        val_mae_hist.append(avg_val_mae)
        epoch_durations.append(time.time() - epoch_start)

        if PRINT_EPOCH_SUMMARY:
            print(f"Fold {fold_idx} Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, train_mae={avg_train_mae:.6f}, val_mae={avg_val_mae:.6f}")

        # early stopping
        if avg_val_loss < best_val_loss - 1e-9:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, os.path.join(MODELS_ROOT, MODEL_NAME, f"best_fold{fold_idx}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[INFO] Fold {fold_idx} early stopping at epoch {epoch}")
                break

    fold_train_time = time.time() - fold_train_start

    # load best weights
    if best_state is not None:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

    model.eval()
    # warmup and latency measure
    with torch.no_grad():
        warm_x = torch.tensor(X_test[:min(BATCH_INFER_WARMUP, X_test.shape[0])], dtype=torch.float32).to(DEVICE)
        _ = model(warm_x)

    iters = 10
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            xb0 = torch.tensor(X_test[:min(BATCH_INFER_WARMUP, X_test.shape[0])], dtype=torch.float32).to(DEVICE)
            _ = model(xb0)
    latency_ms = (time.time() - t0) / (iters * min(BATCH_INFER_WARMUP, X_test.shape[0])) * 1000.0

    # final predictions on test set
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for xb, yb in DataLoader(test_ds, batch_size=BATCH_SIZE):
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            y_pred_list.append(out)
            y_true_list.append(yb.numpy())
    y_pred = np.vstack(y_pred_list)
    y_true = np.vstack(y_true_list)

    # regression metrics
    reg_metrics = {}
    for i, name in enumerate(['Valence','Arousal']):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        reg_metrics[f'{name}_MSE'] = float(mean_squared_error(y_t, y_p))
        reg_metrics[f'{name}_MAE'] = float(mean_absolute_error(y_t, y_p))
        reg_metrics[f'{name}_RMSE'] = float(rmse(y_t, y_p))
        reg_metrics[f'{name}_R2'] = float(r2_score(y_t, y_p))
        reg_metrics[f'{name}_Pearson'] = float(pearsonr(y_t, y_p)[0]) if len(y_t) > 1 else 0.0
        reg_metrics[f'{name}_CCC'] = float(concordance_ccc(y_t, y_p))

    # classification: quadrants using train medians
    def to_quadrant(y_cont, vthr, athr):
        v = (y_cont[:,0] >= vthr).astype(int)
        a = (y_cont[:,1] >= athr).astype(int)
        return (v * 2 + a)

    y_true_quad = to_quadrant(y_true, val_thr, aro_thr)
    y_pred_quad = to_quadrant(y_pred, val_thr, aro_thr)

    class_metrics = {}
    class_metrics['Accuracy'] = float(accuracy_score(y_true_quad, y_pred_quad))
    class_metrics['Precision_macro'] = float(precision_score(y_true_quad, y_pred_quad, average='macro', zero_division=0))
    class_metrics['Recall_macro'] = float(recall_score(y_true_quad, y_pred_quad, average='macro', zero_division=0))
    class_metrics['F1_macro'] = float(f1_score(y_true_quad, y_pred_quad, average='macro', zero_division=0))
    class_metrics['Precision_weighted'] = float(precision_score(y_true_quad, y_pred_quad, average='weighted', zero_division=0))
    class_metrics['Recall_weighted'] = float(recall_score(y_true_quad, y_pred_quad, average='weighted', zero_division=0))
    class_metrics['F1_weighted'] = float(f1_score(y_true_quad, y_pred_quad, average='weighted', zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(y_true_quad, y_pred_quad, labels=[0,1,2,3], zero_division=0)
    per_class = {
        f'class_{i}': {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])}
        for i in range(4)
    }

    # confusion matrices
    cm_raw = confusion_matrix(y_true_quad, y_pred_quad, labels=[0,1,2,3])
    cm_norm = cm_raw.astype('float') / (cm_raw.sum(axis=1)[:, None] + 1e-12)

    # ROC per class using distance-to-center scoring (proxy)
    centers = {
        0: np.array([val_thr - 1.0, aro_thr - 1.0]),
        1: np.array([val_thr - 1.0, aro_thr + 1.0]),
        2: np.array([val_thr + 1.0, aro_thr - 1.0]),
        3: np.array([val_thr + 1.0, aro_thr + 1.0])
    }
    class_scores = np.zeros((y_pred.shape[0], 4), dtype=float)
    for c in range(4):
        d = np.linalg.norm(y_pred - centers[c], axis=1)
        class_scores[:, c] = -d
    roc_auc_per_class = {}
    for c in range(4):
        try:
            y_true_bin = (y_true_quad == c).astype(int)
            y_score = class_scores[:, c]
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc_per_class[f'AUC_class_{c}'] = float(auc(fpr, tpr))
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc_per_class[f'AUC_class_{c}']:.3f}")
            plt.plot([0,1],[0,1],'k--')
            plt.title(f"ROC - Fold{fold_idx} Class{c}")
            plt.xlabel("FPR"); plt.ylabel("TPR")
            savefig(os.path.join(PLOTS_DIR, f"roc_fold{fold_idx}_class{c}.png"))
        except Exception:
            roc_auc_per_class[f'AUC_class_{c}'] = np.nan

    # PR curves per class
    for c in range(4):
        try:
            prec_vals, rec_vals, _ = precision_recall_curve((y_true_quad == c).astype(int), class_scores[:, c])
            plt.figure()
            plt.plot(rec_vals, prec_vals)
            plt.title(f"PR Curve - Fold{fold_idx} Class{c}")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            savefig(os.path.join(PLOTS_DIR, f"pr_fold{fold_idx}_class{c}.png"))
        except Exception:
            pass

    # Save per-fold metrics CSV
    fold_metrics = {
        'fold': fold_idx,
        'param_count': param_count,
        'train_time_s': fold_train_time,
        'train_time_per_epoch_s_mean': float(np.mean(epoch_durations)) if epoch_durations else np.nan,
        'inference_latency_ms_per_sample': float(latency_ms)
    }
    fold_metrics.update(reg_metrics)
    fold_metrics.update(class_metrics)
    fold_metrics.update(roc_auc_per_class)
    per_fold_metrics.append(fold_metrics)

    pd.DataFrame([fold_metrics]).to_csv(os.path.join(METRICS_DIR, f"fold{fold_idx}_summary.csv"), index=False)
    pd.DataFrame(per_class).to_csv(os.path.join(METRICS_DIR, f"fold{fold_idx}_per_class.csv"))
    pd.DataFrame(cm_raw).to_csv(os.path.join(METRICS_DIR, f"fold{fold_idx}_confusion_raw.csv"), index=False)
    pd.DataFrame(cm_norm).to_csv(os.path.join(METRICS_DIR, f"fold{fold_idx}_confusion_norm.csv"), index=False)

    # Save plots: confusion, loss, mae, preds/residuals, correlation
    plt.figure(figsize=(6,5)); sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues'); plt.title(f"Confusion Raw Fold{fold_idx}"); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_confusion_raw.png"))
    plt.figure(figsize=(6,5)); sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues'); plt.title(f"Confusion Norm Fold{fold_idx}"); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_confusion_norm.png"))

    plt.figure(); plt.plot(train_loss_hist, label='train_loss'); plt.plot(val_loss_hist, label='val_loss'); plt.title(f"Loss Fold{fold_idx}"); plt.legend(); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_loss.png"))
    plt.figure(); plt.plot(train_mae_hist, label='train_mae'); plt.plot(val_mae_hist, label='val_mae'); plt.title(f"MAE Fold{fold_idx}"); plt.legend(); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_mae.png"))

    for i, name in enumerate(['Valence','Arousal']):
        plt.figure(figsize=(6,6)); plt.scatter(y_true[:,i], y_pred[:,i], s=10, alpha=0.6)
        mn = min(y_true[:,i].min(), y_pred[:,i].min()); mx = max(y_true[:,i].max(), y_pred[:,i].max())
        plt.plot([mn,mx],[mn,mx],'r--'); plt.title(f"{name} Pred vs Actual Fold{fold_idx}"); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_{name}_pred_vs_actual.png"))
        residuals = y_true[:,i] - y_pred[:,i]
        plt.figure(); sns.histplot(residuals, bins=40, kde=True); plt.title(f"{name} Residuals Fold{fold_idx}"); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_{name}_residuals.png"))

    # correlation heatmap for features vs targets
    if X.shape[1] == len(feature_cols):
        dfcorr = pd.DataFrame(np.hstack([X_scaled, Y]), columns=[*feature_cols, 'Valence','Arousal'])
        plt.figure(figsize=(12,10)); sns.heatmap(dfcorr.corr(), cmap='coolwarm', center=0); savefig(os.path.join(PLOTS_DIR, f"fold{fold_idx}_feature_target_corr.png"))

    print(f"[INFO] Fold {fold_idx} completed and saved.")

# end folds
end_time_all = time.time()
print(f"[INFO] All folds done in {(end_time_all - start_time_all)/60.0:.2f} minutes")

# ---------------------------
# 5. Summary across folds
# ---------------------------
df_folds = pd.DataFrame(per_fold_metrics)
df_folds.to_csv(os.path.join(RESULTS_DIR, "per_fold_metrics.csv"), index=False)
summary = df_folds.describe().transpose()[['mean','std','min','max']]
summary.to_csv(os.path.join(RESULTS_DIR, "folds_summary_stats.csv"))

# choose best fold by lowest total MSE
df_folds['mse_sum'] = df_folds['Valence_MSE'] + df_folds['Arousal_MSE']
best_index = df_folds['mse_sum'].idxmin()
best_fold = int(df_folds.loc[best_index, 'fold'])
src = os.path.join(MODELS_ROOT, MODEL_NAME, f"best_fold{best_fold}.pth")
dst = os.path.join(MODELS_ROOT, MODEL_NAME, f"{MODEL_NAME}_best_overall_fold{best_fold}.pth")
if os.path.exists(src):
    shutil.copy(src, dst)
    print(f"[INFO] Best model copied to {dst}")

# save scaler
joblib.dump(scaler, os.path.join(MODELS_ROOT, MODEL_NAME, "scaler.save"))

print("[INFO] Pipeline complete. Results and models saved at:")
print(" - Results:", RESULTS_DIR)
print(" - Models:", os.path.join(MODELS_ROOT, MODEL_NAME))