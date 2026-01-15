# rel/predict.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Hipe2026SampledPairs, pad_collate
from .model import PersonPlaceRelClassifier, RelBatch, ID2LABEL


def _doc_id(row: Dict[str, Any]) -> str:
    return str(row.get("document_id") or row.get("doc_id") or row.get("id") or "")


def _pair_key(pair: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(pair.get("pers_entity_id") or ""),
        str(pair.get("loc_entity_id") or ""),
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pt")
    ap.add_argument("--input", required=True, help="Input JSONL (dev/test)")
    ap.add_argument("--output", required=True, help="Output predictions JSONL")
    ap.add_argument("--backbone", default="xlm-roberta-base")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--window", type=int, default=1800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1) Load original documents (to preserve required schema fields like `media`)
    docs: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            # Strip whitespace + common invisible chars
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            did = _doc_id(row)
            if did:
                docs[did] = row

    if skipped:
        print(f"[WARN] Skipped {skipped} non-JSON lines in {args.input}")

    if not docs:
        raise RuntimeError("No documents loaded from input JSONL.")

    # 2) Build dataset for prediction (generates one example per sampled_pair)
    ds = Hipe2026SampledPairs(
        args.input,
        tokenizer_name=args.backbone,
        max_length=args.max_length,
        window=args.window,
        negative_downsample=1.0,
        include_unlabeled=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    # 3) Load model
    model = PersonPlaceRelClassifier(backbone=args.backbone)
    model.encoder.resize_token_embeddings(len(ds.tokenizer))
    sd = torch.load(args.model, map_location="cpu")
    model.load_state_dict(sd)
    model.to(args.device)
    model.eval()

    # 4) Predict and store per (doc_id, pers_entity_id, loc_entity_id)
    pred_map: Dict[Tuple[str, str, str], Tuple[str, str]] = {}

    for b in tqdm(dl, desc="predict"):
        input_ids = b["input_ids"].to(args.device)
        attention_mask = b["attention_mask"].to(args.device)

        logits = model(RelBatch(input_ids=input_ids, attention_mask=attention_mask))
        pred_at = torch.argmax(logits["logits_at"], dim=-1).cpu().tolist()
        pred_is = torch.argmax(logits["logits_isat"], dim=-1).cpu().tolist()

        for meta, a, i in zip(b["metas"], pred_at, pred_is):
            (
                document_id,
                pers_entity_id,
                loc_entity_id,
                language,
                date,
                pers_mentions_list,
                loc_mentions_list,
            ) = meta
            pred_map[(document_id, pers_entity_id, loc_entity_id)] = (ID2LABEL[a], ID2LABEL[i])

    # 5) Write output JSONL: same docs, overwrite sampled_pairs labels
    with open(args.output, "w", encoding="utf-8") as f:
        for did, row in docs.items():
            pairs = row.get("sampled_pairs")
            if isinstance(pairs, list):
                for p in pairs:
                    if not isinstance(p, dict):
                        continue
                    pk = (did, str(p.get("pers_entity_id") or ""), str(p.get("loc_entity_id") or ""))
                    if pk in pred_map:
                        at_lbl, is_lbl = pred_map[pk]
                        p["at"] = at_lbl
                        p["isAt"] = is_lbl
                        # explanations can remain null (schema allows null in your sample)
                        if "at_explanation" not in p:
                            p["at_explanation"] = None
                        if "isAt_explanation" not in p:
                            p["isAt_explanation"] = None

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

