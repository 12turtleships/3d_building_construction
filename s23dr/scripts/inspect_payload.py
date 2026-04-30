#!/usr/bin/env python3
"""Inspect internal files in one S23DR payload row."""

from __future__ import annotations

import argparse
import io
import zipfile

import numpy as np
from datasets import load_dataset


def describe_npy(raw: bytes) -> str:
    """Return shape/dtype summary for a .npy blob."""
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # pragma: no cover - best effort metadata
        return f"npy parse error: {exc}"
    return f"shape={arr.shape} dtype={arr.dtype}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="usm3d/s23dr-2026-sampled_4096_v2",
        help="Hugging Face dataset id.",
    )
    p.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation"),
        help="Dataset split.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Example index within split (ignored in streaming mode).",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (faster startup, sequential access).",
    )
    args = p.parse_args()

    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        row = next(iter(ds))
    else:
        ds = load_dataset(args.dataset, split=args.split)
        row = ds[args.index]

    order_id = row.get("order_id", "")
    blob = row.get("data", b"")
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(f"Expected bytes in `data`, got {type(blob).__name__}")

    print(f"dataset={args.dataset} split={args.split} order_id={order_id}")
    print(f"payload_bytes={len(blob)} zip_magic={bytes(blob[:4])!r}")
    print("-" * 72)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = zf.infolist()
        print(f"zip_members={len(members)}")
        for info in members:
            line = f"{info.filename} ({info.file_size} bytes)"
            if info.filename.endswith(".npy"):
                raw = zf.read(info.filename)
                line += f" -> {describe_npy(raw)}"
            print(line)


if __name__ == "__main__":
    main()
