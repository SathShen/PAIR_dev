"""PAIR dataset package."""

from .pair_dataset import (
    DatasetSpec,
    CanonicalChangeTarget,
    UnifiedPAIRDataset,
    UNKNOWN_CLASS_ID,
    IGNORE_CLASS_ID,
    infer_unchanged_raw_id,
    infer_binary_class_ids,
)
