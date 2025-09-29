#!/usr/bin/env python3
"""
Final NOTAM Subject Classifier — LoRA on DeBERTa-v3-large (fp16), Windows-safe.

Key choices:
- Encoder: microsoft/deberta-v3-large (very strong for NLU classification)
- Tokenizer: use_fast=False  ← avoids Windows+tiktoken conversion error
- Imbalance: Focal Loss (gamma) + class-weight alpha + WeightedRandomSampler
- Splits: keep singleton labels entirely in train; stratify the rest
- Context: max_len=512; grad-accum + (optional) grad checkpointing for 3050 GPU
- Metrics: Top-K, macro averages, calibrated threshold with coverage floor
- Artifacts: adapters + merged model (safetensors), label encoder/map, history, meta, logs

Run:
  python train_subject_final_debv3_lora.py --csv "E:/E DRIVE DATA/NOTAM RAG pipeline/Main RAG Pipeline/Notam_subject.csv"
"""

import os, re, json, time, argparse, logging, random
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

# ------------------------- Logging -------------------------
def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

CREATED_SRC_RE = re.compile(r"CREATED:\s.*?SOURCE:\s*[A-Z0-9]+", re.IGNORECASE | re.DOTALL)

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = CREATED_SRC_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------- Dataset -------------------------
class NotamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.labels = np.array(labels, dtype=np.int64)
        enc = tokenizer(list(texts), truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ------------------------- Eval / Metrics -------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    logits_all, labels_all = [], []
    for batch in dataloader:
        b = {k: v.to(device) for k, v in batch.items()}
        labels = b.pop("labels")
        out = model(**b)
        logits_all.append(out.logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    logits = torch.cat(logits_all, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    preds = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }, logits, labels

def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    if logits.size == 0: return 0.0
    k = min(k, logits.shape[1])
    topk_idx = np.argpartition(-logits, kth=k-1, axis=1)[:, :k]
    hits = np.any(topk_idx == labels.reshape(-1, 1), axis=1)
    return float(hits.mean())

def choose_threshold(probs: np.ndarray, labels: np.ndarray,
                     coverage_floor=0.40, optimize="macro_f1"):
    best = {"t": 0.5, "coverage": 0.0, "macro_f1": -1.0,
            "accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}
    grid = np.linspace(0.20, 0.95, 76)
    for t in grid:
        max_i = probs.argmax(axis=1)
        max_p = probs[np.arange(len(probs)), max_i]
        pred  = np.where(max_p >= t, max_i, -1)
        mask  = pred != -1
        coverage = float(mask.mean())
        if coverage < coverage_floor or not mask.any():
            continue
        f1   = f1_score(labels[mask], pred[mask], average="macro", zero_division=0)
        acc  = accuracy_score(labels[mask], pred[mask])
        prec = precision_score(labels[mask], pred[mask], average="macro", zero_division=0)
        rec  = recall_score(labels[mask], pred[mask], average="macro", zero_division=0)
        key  = f1 if optimize == "macro_f1" else prec
        best_key = best["macro_f1"] if optimize == "macro_f1" else best["macro_precision"]
        if key > best_key:
            best = {"t": float(t), "coverage": coverage, "macro_f1": float(f1),
                    "accuracy": float(acc), "macro_precision": float(prec), "macro_recall": float(rec)}
    if best["macro_f1"] < 0:  # predict-all fallback
        t = 0.0
        max_i = probs.argmax(axis=1)
        pred  = max_i
        f1   = f1_score(labels, pred, average="macro", zero_division=0)
        acc  = accuracy_score(labels, pred)
        prec = precision_score(labels, pred, average="macro", zero_division=0)
        rec  = recall_score(labels, pred, average="macro", zero_division=0)
        best = {"t": t, "coverage": 1.0, "macro_f1": float(f1), "accuracy": float(acc),
                "macro_precision": float(prec), "macro_recall": float(rec)}
    return best

def robust_splits(X: pd.Series, y: pd.Series, seed=42, test_size=0.15, log=None) -> Tuple[pd.Series, ...]:
    """Keep labels with <2 samples entirely in train; stratify the rest."""
    X = pd.Series(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    counts = y.value_counts()
    rare = counts[counts < 2].index.tolist()

    idx_all = np.arange(len(y))
    idx_rare = idx_all[y.isin(rare)]
    idx_rest = idx_all[~y.isin(rare)]

    X_train_list = [X.iloc[idx_rare]]
    y_train_list = [y.iloc[idx_rare]]

    if len(idx_rest) > 0:
        X_rest, y_rest = X.iloc[idx_rest], y.iloc[idx_rest]
        X_tr_nr, X_tmp, y_tr_nr, y_tmp = train_test_split(
            X_rest, y_rest, test_size=test_size, random_state=seed, stratify=y_rest
        )
        X_train_list.append(X_tr_nr); y_train_list.append(y_tr_nr)
        try:
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
            )
        except Exception as e:
            if log: log.warning(f"Stratified val/test split failed ({e}); using unstratified split.")
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=None
            )
    else:
        X_val = pd.Series([], dtype=object); y_val = pd.Series([], dtype=np.int64)
        X_te  = pd.Series([], dtype=object); y_te  = pd.Series([], dtype=np.int64)

    X_tr = pd.concat(X_train_list).reset_index(drop=True)
    y_tr = pd.concat(y_train_list).reset_index(drop=True)
    if log: log.info(f"Split summary → train={len(X_tr)}, val={len(X_val)}, test={len(X_te)} (rare-only-in-train={len(idx_rare)})")
    return X_tr, X_val, X_te, y_tr, y_val, y_te

# ------------------------- Loss -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha_vec: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha_vec", alpha_vec if alpha_vec is not None else torch.ones(num_classes))
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        pt = probs[range(targets.size(0)), targets]
        alpha = self.alpha_vec[targets]
        loss = -alpha * ((1 - pt) ** self.gamma) * log_probs[range(targets.size(0)), targets]
        return loss.mean()

# ------------------------- LoRA targets ----------------------
def discover_deberta_targets(model) -> List[str]:
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    picks = set()
    for k in ["query_proj", "key_proj", "value_proj", "q_proj", "k_proj", "v_proj"]:
        if any(k in n for n in names): picks.add(k)
    for k in ["query", "key", "value"]:
        if any(re.search(rf"\b{k}\b", n) for n in names): picks.add(k)
    if any("dense" in n for n in names): picks.add("dense")
    if not picks:
        picks = {"query_proj", "key_proj", "value_proj", "dense"}
    return sorted(picks)

def rebuild_classifier_fp16(model_name: str, base, num_labels: int, device):
    head = None
    try:
        if "deberta" in model_name.lower():
            from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ClassificationHead
            head = DebertaV2ClassificationHead(base.config)
        elif "roberta" in model_name.lower():
            from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
            head = RobertaClassificationHead(base.config)
    except Exception:
        head = None
    if head is None:
        head = nn.Sequential(
            nn.Dropout(getattr(base.config, "hidden_dropout_prob", 0.1)),
            nn.Linear(base.config.hidden_size, num_labels)
        )
    # ensure output size
    if isinstance(head, nn.Sequential):
        out = head[-1]
        if isinstance(out, nn.Linear) and out.out_features != num_labels:
            head[-1] = nn.Linear(out.in_features, num_labels)
    else:
        if hasattr(head, "out_proj") and isinstance(head.out_proj, nn.Linear):
            if head.out_proj.out_features != num_labels:
                head.out_proj = nn.Linear(head.out_proj.in_features, num_labels)

    head.to(device).to(dtype=torch.float16)
    base.classifier = head
    return base

def build_model(model_name: str, num_labels: int, device, lora_r: int, lora_alpha: int, use_grad_ckpt: bool):
    # Force slow tokenizer (Windows safe)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Force safetensors path
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,   # we'll move to CUDA right after
    )

    if use_grad_ckpt:
        try: base.gradient_checkpointing_enable()
        except Exception: pass

    base = rebuild_classifier_fp16(model_name, base, num_labels, device)

    targets = discover_deberta_targets(base)
    log.info(f"LoRA targets: {targets}")

    lconf = LoraConfig(
        r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=0.1,
        target_modules=targets, task_type="SEQ_CLS", modules_to_save=["classifier"],
    )
    model = get_peft_model(base, lconf).to(device)
    model.print_trainable_parameters()
    return model, tokenizer

# ------------------------- Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--csv", required=True, help="Path to Notam_subject.csv")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--run-name", default=None)

    # data/inputs
    ap.add_argument("--text-col", default="combined_text")
    ap.add_argument("--max-len", type=int, default=512)

    # model
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")

    # training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4, help="Per-step batch size (small by default for 3050 GPU)")
    ap.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=4)

    # loss & thresholding
    ap.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    ap.add_argument("--coverage-floor", type=float, default=0.50)
    ap.add_argument("--opt-threshold-for", choices=["macro_f1", "precision"], default="macro_f1")

    args = ap.parse_args()

    # preflight
    try:
        import sentencepiece  # noqa
    except Exception as e:
        raise RuntimeError("Install sentencepiece: pip install sentencepiece") from e

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{Path(args.model).name.replace('/','_')}_r{args.lora_r}_a{args.lora_alpha}_l{args.max_len}_focal"
    out_dir = Path(args.outdir) / run_name
    (out_dir / "adapters").mkdir(parents=True, exist_ok=True)
    (out_dir / "merged").mkdir(parents=True, exist_ok=True)
    global log
    log = make_logger(out_dir / "train.log")

    log.info("Environment check...")
    log.info(f"torch={torch.__version__}")
    try:
        import transformers as _tf; import peft as _pf
        log.info(f"transformers={_tf.__version__} | peft={_pf.__version__}")
    except Exception:
        pass

    set_all_seeds(args.seed)
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("medium")
    except Exception: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # ---------------- Load & Preprocess ----------------
    log.info("Loading CSV...")
    df = pd.read_csv(args.csv)

    if "combined_text" not in df.columns:
        df["combined_text"] = (
            df.get("E_field_original", "").astype(str).fillna("") + " [SEP] " +
            df.get("E_field_expanded", "").astype(str).fillna("")
        ).map(clean_text)
    else:
        df["combined_text"] = df["combined_text"].astype(str).map(clean_text)

    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not in CSV; available: {list(df.columns)}")
    if "Subject" not in df.columns:
        raise ValueError("CSV must contain 'Subject' column.")

    df = df[(df[args.text_col].astype(str).str.len() > 0) & (df["Subject"].astype(str).str.len() > 0)].copy()
    df["Subject"] = df["Subject"].astype(str).str.strip()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Subject"])
    classes = le.classes_.tolist()
    num_labels = len(classes)
    log.info(f"Data loaded: {len(df)} rows, {num_labels} classes.")

    X = df[args.text_col]
    y = df["label"]

    # ---------- Splits ----------
    X_tr, X_val, X_te, y_tr, y_val, y_te = robust_splits(X, y, seed=args.seed, test_size=0.15, log=log)
    if len(X_val) == 0 or len(X_te) == 0:
        log.warning("Validation or Test set ended up empty due to extreme class imbalance.")

    # ---------- Build model & tokenizer ----------
    model, tokenizer = build_model(args.model, num_labels, device, args.lora_r, args.lora_alpha, use_grad_ckpt=args.grad_ckpt)

    # ---------- Datasets & loaders ----------
    ds_tr  = NotamDataset(X_tr.tolist(),  y_tr.tolist(),  tokenizer, args.max_len)
    ds_val = NotamDataset(X_val.tolist(), y_val.tolist(), tokenizer, args.max_len) if len(X_val) else None
    ds_te  = NotamDataset(X_te.tolist(),  y_te.tolist(),  tokenizer, args.max_len) if len(X_te) else None

    counts = Counter(y_tr.tolist())
    weights = np.array([1.0 / counts[yi] for yi in y_tr.tolist()], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    pin_mem = torch.cuda.is_available()
    dl_tr  = DataLoader(ds_tr,  batch_size=args.batch, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=pin_mem)
    dl_val = DataLoader(ds_val, batch_size=max(1,args.batch), shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin_mem) if ds_val else None
    dl_te  = DataLoader(ds_te,  batch_size=max(1,args.batch), shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin_mem) if ds_te else None

    # ---------- Loss / Optim / Sched ----------
    present = np.unique(np.array(y_tr))
    w_present = compute_class_weight(class_weight="balanced", classes=present, y=np.array(y_tr))
    w_vec = np.ones(num_labels, dtype=np.float32); w_vec[present] = w_present
    w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=device)

    alpha_vec = w_tensor.clone(); alpha_vec = alpha_vec / alpha_vec.mean()
    loss_fn = FocalLoss(num_classes=num_labels, gamma=args.gamma, alpha_vec=alpha_vec)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dl_tr) // max(1, args.accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ---------- Train loop with early stop ----------
    best_val = -1.0
    bad_epochs = 0
    patience = int(args.patience)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        step = 0
        accum = max(1, args.accum)

        for i, batch in enumerate(dl_tr):
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = b.pop("labels")
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(**b)
                loss = loss_fn(out.logits, labels) / accum
            scaler.scale(loss).backward()

            if ((i + 1) % accum == 0) or ((i + 1) == len(dl_tr)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

            total_loss += loss.item()

        tr_loss = total_loss / max(1, len(dl_tr))

        if dl_val is not None:
            val_metrics, val_logits_tmp, val_labels_tmp = evaluate(model, dl_val, device)
            val_top3 = topk_accuracy(val_logits_tmp, val_labels_tmp, k=3)
            val_top5 = topk_accuracy(val_logits_tmp, val_labels_tmp, k=5)
            epoch_time = time.time() - t0
            gpu_mem = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
            log.info(
                f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"Top3={val_top3:.3f} | Top5={val_top5:.3f} | time={epoch_time:.1f}s | GPU_peak={gpu_mem:.2f}GB | steps={step}"
            )

            history.append({
                "epoch": epoch, "train_loss": tr_loss,
                "val_acc": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                "val_top3": val_top3, "val_top5": val_top5, "epoch_s": epoch_time, "gpu_peak_gb": gpu_mem,
                "steps": step
            })

            if val_metrics["macro_f1"] > best_val:
                best_val, bad_epochs = val_metrics["macro_f1"], 0
                model.save_pretrained(str(out_dir / "adapters"), safe_serialization=True)
                tokenizer.save_pretrained(str(out_dir / "adapters"))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log.info("Early stopping.")
                    break
        else:
            epoch_time = time.time() - t0
            log.info(f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} | time={epoch_time:.1f}s")
            history.append({"epoch": epoch, "train_loss": tr_loss, "epoch_s": epoch_time})
            model.save_pretrained(str(out_dir / "adapters"), safe_serialization=True)
            tokenizer.save_pretrained(str(out_dir / "adapters"))

    # --------- Merge adapters into base ----------
    base_for_merge = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, use_safetensors=True, low_cpu_mem_usage=True
    )
    base_for_merge = rebuild_classifier_fp16(args.model, base_for_merge, num_labels, device)
    merged = PeftModel.from_pretrained(base_for_merge, str(out_dir / "adapters"))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir / "merged"))

    # ---------------- Threshold selection & evaluation ----------------
    merged.to(device)

    if dl_val is not None:
        val_metrics, val_logits, val_labels = evaluate(merged, dl_val, device)
        val_probs = torch.tensor(val_logits).softmax(dim=-1).numpy()
        thr = choose_threshold(val_probs, val_labels, coverage_floor=args.coverage_floor,
                               optimize=args.opt_threshold_for)
        val_top3 = topk_accuracy(val_logits, val_labels, k=3)
        val_top5 = topk_accuracy(val_logits, val_labels, k=5)
        log.info(f"Validation (merged) Top-3: {val_top3:.3f} | Top-5: {val_top5:.3f}")
        try:
            preds = val_logits.argmax(axis=1)
            rep = classification_report(val_labels, preds, zero_division=0)
            log.info("Validation classification report (top-1):\n" + rep)
        except Exception:
            pass
    else:
        val_metrics = {"accuracy": None, "macro_f1": None, "macro_precision": None, "macro_recall": None}
        thr = {"t": 0.5, "coverage": None, "macro_f1": None, "accuracy": None,
               "macro_precision": None, "macro_recall": None}
        val_top3 = val_top5 = None

    if dl_te is not None:
        te_metrics, te_logits, te_labels = evaluate(merged, dl_te, device)
        te_top3 = topk_accuracy(te_logits, te_labels, k=3)
        te_top5 = topk_accuracy(te_logits, te_labels, k=5)
        log.info(f"Test Top-3: {te_top3:.3f} | Top-5: {te_top5:.3f}")

        te_probs = torch.tensor(te_logits).softmax(dim=-1).numpy()
        te_max_i = te_probs.argmax(axis=1)
        te_max_p = te_probs[np.arange(len(te_probs)), te_max_i]
        te_pred  = np.where(te_max_p >= float(thr["t"]), te_max_i, -1)
        covered = te_pred != -1
        test_stats = {"coverage": float(covered.mean())}
        if covered.any():
            test_stats.update({
                "accuracy": float(accuracy_score(te_labels[covered], te_pred[covered])),
                "macro_f1": float(f1_score(te_labels[covered], te_pred[covered], average="macro", zero_division=0)),
                "macro_precision": float(precision_score(te_labels[covered], te_pred[covered], average="macro", zero_division=0)),
                "macro_recall": float(recall_score(te_labels[covered], te_pred[covered], average="macro", zero_division=0)),
            })
        else:
            test_stats.update({"accuracy": 0.0, "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0})
        try:
            top1 = te_logits.argmax(axis=1)
            rep = classification_report(te_labels, top1, zero_division=0)
            log.info("Test classification report (top-1):\n" + rep)
        except Exception:
            pass
    else:
        test_stats = {"coverage": None, "accuracy": None, "macro_f1": None,
                      "macro_precision": None, "macro_recall": None}
        te_top3 = te_top5 = None

    # ---------------- Save metadata, label map, history ----------------
    dump(le, str(out_dir / "label_encoder.joblib"))
    (out_dir / "label_map.json").write_text(json.dumps({i: c for i, c in enumerate(classes)}, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    meta = {
        "base_model": args.model,
        "lora": True,
        "adapters_dir": str(out_dir / "adapters"),
        "merged_dir": str(out_dir / "merged"),
        "threshold": float(thr["t"]),
        "coverage_floor": float(args.coverage_floor),
        "opt_threshold_for": args.opt_threshold_for,
        "classes": classes,
        "text_col": args.text_col,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch": args.batch,
        "accum": args.accum,
        "lr": args.lr,
        "valid_metrics_raw": val_metrics,
        "valid_metrics_thresholded": thr,
        "valid_topk": {"top3": val_top3, "top5": val_top5},
        "test_metrics": test_stats,
        "test_topk": {"top3": te_top3, "top5": te_top5},
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "grad_ckpt": bool(args.grad_ckpt),
        "run_name": run_name,
    }
    (out_dir / "subject_clf_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Training complete.")
    log.info(f"Validation (no threshold): {val_metrics}")
    log.info(f"Validation (threshold/meta): {thr}")
    log.info(f"Test (covered): {test_stats}")
    log.info(f"Saved adapters -> {out_dir/'adapters'}")
    log.info(f"Saved merged   -> {out_dir/'merged'}")
    log.info(f"Saved label encoder -> {out_dir/'label_encoder.joblib'}")
    log.info(f"Saved meta -> {out_dir/'subject_clf_meta.json'}")

if __name__ == "__main__":
    main()
