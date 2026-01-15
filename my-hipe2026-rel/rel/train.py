# rel/train.py
from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import Hipe2026SampledPairs, pad_collate
from .model import PersonPlaceRelClassifier, RelBatch, LABELS


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recalls = []
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(recalls))


@torch.no_grad()
def evaluate(model: PersonPlaceRelClassifier, dl: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    y_at_t, y_at_p = [], []
    y_is_t, y_is_p = [], []

    for b in dl:
        input_ids = b["input_ids"].to(device)
        attention_mask = b["attention_mask"].to(device)
        labels_at = b["labels_at"].to(device)
        labels_isat = b["labels_isat"].to(device)

        logits = model(RelBatch(input_ids=input_ids, attention_mask=attention_mask))
        pred_at = torch.argmax(logits["logits_at"], dim=-1)
        pred_is = torch.argmax(logits["logits_isat"], dim=-1)

        y_at_t.extend(labels_at.cpu().tolist())
        y_at_p.extend(pred_at.cpu().tolist())
        y_is_t.extend(labels_isat.cpu().tolist())
        y_is_p.extend(pred_is.cpu().tolist())

    y_at_t = np.array(y_at_t, dtype=np.int64)
    y_at_p = np.array(y_at_p, dtype=np.int64)
    y_is_t = np.array(y_is_t, dtype=np.int64)
    y_is_p = np.array(y_is_p, dtype=np.int64)

    print("\n[DEV] at classification report:")
    print(classification_report(
                    y_at_t, y_at_p,
                    labels=[0, 1, 2],
                    target_names=LABELS,
                    digits=4,
                    zero_division=0
                    ))
    print("[DEV] isAt classification report:")
    print(classification_report(
    y_is_t, y_is_p,
    labels=[0, 1, 2],
    target_names=LABELS,
    digits=4,
    zero_division=0
    ))  

    m_at = macro_recall(y_at_t, y_at_p)
    m_is = macro_recall(y_is_t, y_is_p)

    return {
        "at_macro_recall": m_at,
        "isat_macro_recall": m_is,
        "avg_macro_recall": 0.5 * (m_at + m_is),
        "at_acc": float((y_at_t == y_at_p).mean()),
        "isat_acc": float((y_is_t == y_is_p).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train jsonl")
    ap.add_argument("--dev", required=True, help="dev jsonl")
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--window", type=int, default=1800)
    ap.add_argument("--negative_downsample", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train_ds = Hipe2026SampledPairs(
        args.train,
        tokenizer_name=args.backbone,
        max_length=args.max_length,
        window=args.window,
        negative_downsample=args.negative_downsample,
        include_unlabeled=False,  # train expects labels
    )
    dev_ds = Hipe2026SampledPairs(
        args.dev,
        tokenizer_name=args.backbone,
        max_length=args.max_length,
        window=args.window,
        negative_downsample=1.0,
        include_unlabeled=False,  # dev expects labels
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    # Gather labels to build class weights
    y_at, y_is = [], []
    for i in range(len(train_ds)):
        it = train_ds[i]
        y_at.append(int(it["labels_at"]))
        y_is.append(int(it["labels_isat"]))
    y_at = np.array(y_at, dtype=np.int64)
    y_is = np.array(y_is, dtype=np.int64)

    w_at = compute_class_weights(y_at).to(args.device)
    w_is = compute_class_weights(y_is).to(args.device)

    model = PersonPlaceRelClassifier(backbone=args.backbone)
    # resize because we added special marker tokens to tokenizer
    model.encoder.resize_token_embeddings(len(train_ds.tokenizer))
    model.to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(train_dl)
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False)

        for b in pbar:
            optim.zero_grad(set_to_none=True)

            input_ids = b["input_ids"].to(args.device)
            attention_mask = b["attention_mask"].to(args.device)
            labels_at = b["labels_at"].to(args.device)
            labels_isat = b["labels_isat"].to(args.device)

            loss = model.compute_loss(
                RelBatch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels_at=labels_at,
                    labels_isat=labels_isat,
                ),
                class_weights_at=w_at,
                class_weights_isat=w_is,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{running / max(1, (pbar.n + 1)):.4f}")

        print(f"\nEpoch {epoch} train_loss={running / max(1, len(train_dl)):.4f}")
        metrics = evaluate(model, dev_dl, args.device)
        print(f"[DEV] {metrics}")

        if metrics["avg_macro_recall"] > best:
            best = metrics["avg_macro_recall"]
            torch.save(model.state_dict(), args.out)
            print(f"Saved best model to {args.out} (avg_macro_recall={best:.4f})")

    print(f"\nDone. Best avg_macro_recall={best:.4f}")


if __name__ == "__main__":
    main()

