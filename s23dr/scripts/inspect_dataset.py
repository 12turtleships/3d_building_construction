#!/usr/bin/env python3
"""Stream one row from the public S23DR 2026 sampled dataset (metadata sanity check)."""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation"),
        help="Dataset split to peek at.",
    )
    p.add_argument(
        "--dataset",
        default="usm3d/s23dr-2026-sampled_4096_v2",
        help="Hugging Face dataset id.",
    )
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    row = next(iter(ds))
    order_id = row.get("order_id", "")
    blob = row.get("data")
    n = len(blob) if blob is not None else 0
    print(f"dataset={args.dataset} split={args.split}", flush=True)
    print(f"order_id={order_id!r} data_type={type(blob).__name__} data_bytes={n}", flush=True)
    # Streaming can leave non-daemon threads; hard-exit so CI/scripts don't hang.
    os._exit(0)


if __name__ == "__main__":
    main()
