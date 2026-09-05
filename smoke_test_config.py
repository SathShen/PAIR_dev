#!/usr/bin/env python3
"""Preflight PAIR's directory-as-schema config without loading Qwen."""

import argparse
from pathlib import Path

from datasets.config_loader import load_experiment_config
from datasets.pair_dataset import UnifiedPAIRDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pair_train.json"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--load-first-sample",
        action="store_true",
    )
    args = p.parse_args()

    cfg = load_experiment_config(
        args.config,
        args.datasets,
    )

    print("=" * 92)
    print("PAIR DIRECTORY-AS-SCHEMA PREFLIGHT")
    print("=" * 92)
    print("Experiment:", cfg.experiment["name"])
    print("Selected  :", list(cfg.selected_names))
    print()

    for name in cfg.selected_names:
        ds = cfg.datasets[name]

        print(f"[{name}]")
        print("  root               :", ds.root)
        print("  modalities (auto)  :", list(ds.modalities))
        print("  route (auto)       :", ds.route)
        print("  label mode (auto)  :", ds.label_mode)
        print("  unchanged id (auto):", ds.unchanged_raw_id)
        print("  train manifest     :", ds.train_manifest)
        print("  val manifest       :", ds.val_manifest)
        print("  test manifest      :", ds.test_manifest)
        print("  per-GPU batch      :", ds.per_gpu_batch_size)
        print("  classes            :", ds.class_names)

        if args.load_first_sample:
            dataset = UnifiedPAIRDataset(
                ds.train_manifest,
                ds.spec,
            )
            sample = dataset[0]
            print("  first sample id    :", sample["sample_id"])
            print("  sample route       :", sample["route"])
            print("  prompt             :", sample["prompt"])
            print(
                "  target shapes      :",
                {
                    k: tuple(v.shape)
                    for k, v in sample["target"].items()
                },
            )

        print()

    print("CONFIG/DATASET PREFLIGHT PASS")


if __name__ == "__main__":
    main()
