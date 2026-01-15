# rel/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

LABELS = ["FALSE", "PROBABLE", "TRUE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class RelBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels_at: Optional[torch.Tensor] = None
    labels_isat: Optional[torch.Tensor] = None


class PersonPlaceRelClassifier(nn.Module):
    """
    Cross-encoder with two 3-way heads:
      - at:   FALSE / PROBABLE / TRUE
      - isAt: FALSE / PROBABLE / TRUE
    """

    def __init__(self, backbone: str = "xlm-roberta-base", dropout: float = 0.1):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone, config=cfg)
        self.dropout = nn.Dropout(dropout)
        self.head_at = nn.Linear(cfg.hidden_size, 3)
        self.head_isat = nn.Linear(cfg.hidden_size, 3)

    def forward(self, batch: RelBatch) -> Dict[str, torch.Tensor]:
        out = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ).last_hidden_state
        cls = self.dropout(out[:, 0])  # CLS pooling
        return {
            "logits_at": self.head_at(cls),
            "logits_isat": self.head_isat(cls),
        }

    def compute_loss(
        self,
        batch: RelBatch,
        class_weights_at: Optional[torch.Tensor] = None,
        class_weights_isat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert batch.labels_at is not None and batch.labels_isat is not None
        logits = self.forward(batch)

        ce_at = nn.CrossEntropyLoss(weight=class_weights_at)
        ce_isat = nn.CrossEntropyLoss(weight=class_weights_isat)

        loss_at = ce_at(logits["logits_at"], batch.labels_at)
        loss_isat = ce_isat(logits["logits_isat"], batch.labels_isat)
        return loss_at + loss_isat
