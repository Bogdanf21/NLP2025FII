# rel/data.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .model import LABEL2ID

P_START, P_END = "[PERS]", "[/PERS]"
L_START, L_END = "[LOC]", "[/LOC]"


def _norm_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        u = x.strip().upper()
        if u in ("FALSE", "PROBABLE", "TRUE"):
            return u
        if u in ("POSSIBLE", "PLAUSIBLE", "LIKELY"):
            return "PROBABLE"
        if u in ("YES",):
            return "TRUE"
        if u in ("NO", "NONE"):
            return "FALSE"
    return None


def _safe_find(text: str, sub: str) -> Optional[Tuple[int, int]]:
    if not text or not sub:
        return None
    m = re.search(re.escape(sub), text)
    if not m:
        return None
    return m.start(), m.end()


def _insert_markers(text: str, spans: List[Tuple[int, int, str, str]]) -> str:
    # Insert from right to left so offsets remain valid
    spans = sorted(spans, key=lambda x: x[0], reverse=True)
    out = text
    for s, e, st, en in spans:
        if 0 <= s <= e <= len(out):
            out = out[:s] + st + out[s:e] + en + out[e:]
    return out


def build_marked_context(
    text: str,
    pers_mentions: List[str],
    loc_mentions: List[str],
    window: int = 1800,
) -> str:
    """
    Marker-based context extraction.
    Prefer locating the first person mention and first location mention in the doc.
    Falls back to prefixing mentions + doc head if not found.
    """
    p_m = next((m for m in pers_mentions if isinstance(m, str) and m.strip()), "")
    l_m = next((m for m in loc_mentions if isinstance(m, str) and m.strip()), "")

    p_span = _safe_find(text, p_m)
    l_span = _safe_find(text, l_m)

    if p_span and l_span:
        s0 = min(p_span[0], l_span[0])
        e0 = max(p_span[1], l_span[1])
        center = (s0 + e0) // 2
        ws = max(0, center - window // 2)
        we = min(len(text), center + window // 2)
        chunk = text[ws:we]

        # relocate spans inside chunk
        p2 = _safe_find(chunk, p_m)
        l2 = _safe_find(chunk, l_m)
        spans = []
        if p2:
            spans.append((p2[0], p2[1], P_START, P_END))
        if l2:
            spans.append((l2[0], l2[1], L_START, L_END))
        if spans:
            return _insert_markers(chunk, spans)

    # fallback: prefix + head
    prefix = f"{P_START}{p_m}{P_END} {L_START}{l_m}{L_END}\n\n"
    return prefix + text[:window]


@dataclass
class PairExample:
    document_id: str
    language: str
    date: str
    pers_entity_id: str
    loc_entity_id: str
    pers_mentions_list: List[str]
    loc_mentions_list: List[str]
    input_text: str
    y_at: Optional[int]
    y_isat: Optional[int]


class Hipe2026SampledPairs(Dataset):
    """
    HIPE-2026 dataset reader for JSONL where supervision is inside `sampled_pairs`.
    Each row is a document with:
      - document_id, language, date, text, ...
      - sampled_pairs: list of pair dicts with IDs, mention lists, and at/isAt labels (train/dev).
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str = "xlm-roberta-base",
        max_length: int = 384,
        window: int = 1800,
        negative_downsample: float = 1.0,
        seed: int = 13,
        include_unlabeled: bool = True,
    ):
        super().__init__()
        import random

        random.seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [P_START, P_END, L_START, L_END]}
        )
        self.max_length = max_length
        self.items: List[PairExample] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                doc_id = str(row.get("document_id") or row.get("doc_id") or row.get("id") or "")
                lang = str(row.get("language") or "")
                date = str(row.get("date") or row.get("publication_date") or "")
                text = str(row.get("text") or "")

                pairs = row.get("sampled_pairs")
                if not isinstance(pairs, list) or not text or not doc_id:
                    continue

                for p in pairs:
                    if not isinstance(p, dict):
                        continue

                    pers_entity_id = str(p.get("pers_entity_id") or "")
                    loc_entity_id = str(p.get("loc_entity_id") or "")
                    pers_mentions = p.get("pers_mentions_list") or []
                    loc_mentions = p.get("loc_mentions_list") or []
                    if not pers_entity_id or not loc_entity_id:
                        continue
                    if not isinstance(pers_mentions, list):
                        pers_mentions = []
                    if not isinstance(loc_mentions, list):
                        loc_mentions = []

                    at_lbl = _norm_label(p.get("at"))
                    isat_lbl = _norm_label(p.get("isAt"))

                    y_at = LABEL2ID.get(at_lbl) if at_lbl else None
                    y_isat = LABEL2ID.get(isat_lbl) if isat_lbl else None

                    # Keep unlabeled pairs (for test) if include_unlabeled
                    if (y_at is None or y_isat is None) and not include_unlabeled:
                        continue

                    # Optional negative downsampling when labels exist and both are FALSE
                    if (
                        y_at is not None
                        and y_isat is not None
                        and y_at == LABEL2ID["FALSE"]
                        and y_isat == LABEL2ID["FALSE"]
                        and negative_downsample < 1.0
                    ):
                        if random.random() > negative_downsample:
                            continue

                    marked = build_marked_context(text, pers_mentions, loc_mentions, window=window)

                    # Provide date and language as extra signals (cheap but helps)
                    extra = f"\n\nlang={lang}\ndate={date}"
                    input_text = (
                                marked + f"\n\nPAIR: person={pers_mentions[0] if pers_mentions else ''} "
                                f"place={loc_mentions[0] if loc_mentions else ''} "
                                f"date={date}"
                    )
                    self.items.append(
                        PairExample(
                            document_id=doc_id,
                            language=lang,
                            date=date,
                            pers_entity_id=pers_entity_id,
                            loc_entity_id=loc_entity_id,
                            pers_mentions_list=pers_mentions,
                            loc_mentions_list=loc_mentions,
                            input_text=input_text,
                            y_at=y_at,
                            y_isat=y_isat,
                        )
                    )

        if len(self.items) == 0:
            raise RuntimeError(
                "Dataset produced 0 examples. Check that your JSONL contains `sampled_pairs` and `text`."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.items[i]
        enc = self.tokenizer(
            ex.input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        out: Dict[str, Any] = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "meta": (
                ex.document_id,
                ex.pers_entity_id,
                ex.loc_entity_id,
                ex.language,
                ex.date,
                ex.pers_mentions_list,
                ex.loc_mentions_list,
            ),
        }

        if ex.y_at is not None and ex.y_isat is not None:
            out["labels_at"] = torch.tensor(ex.y_at, dtype=torch.long)
            out["labels_isat"] = torch.tensor(ex.y_isat, dtype=torch.long)

        return out


def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)

    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    metas = []
    has_labels = ("labels_at" in batch[0]) and ("labels_isat" in batch[0])

    labels_at = torch.zeros((bsz,), dtype=torch.long) if has_labels else None
    labels_isat = torch.zeros((bsz,), dtype=torch.long) if has_labels else None

    for i, x in enumerate(batch):
        ids = x["input_ids"]
        mask = x["attention_mask"]
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(mask)] = mask
        metas.append(x["meta"])
        if has_labels:
            labels_at[i] = x["labels_at"]
            labels_isat[i] = x["labels_isat"]

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "metas": metas,
    }
    if has_labels:
        out["labels_at"] = labels_at
        out["labels_isat"] = labels_isat
    return out

