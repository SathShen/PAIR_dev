"""PAIR dataset package."""


from .pair_dataset import (
    DatasetSpec,
    CanonicalChangeTarget,
    UnifiedPAIRDataset,
    infer_unchanged_raw_id,
    infer_binary_class_ids,
)