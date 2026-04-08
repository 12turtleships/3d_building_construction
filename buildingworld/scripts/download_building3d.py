#!/usr/bin/env python3
"""Download Building3D from Hugging Face (dataset is gated — see buildingworld/README.md)."""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        default="Building3D/Building3D",
        help="Hugging Face dataset repository id.",
    )
    p.add_argument(
        "--local-dir",
        default="outputs/building3d_hf",
        help="Directory to download files into (created if missing).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF API token (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN).",
    )
    args = p.parse_args()

    if not args.token:
        print(
            "Error: no Hugging Face token. Request dataset access at "
            "https://huggingface.co/datasets/Building3D/Building3D then run "
            "`huggingface-cli login` or export HF_TOKEN=...",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.local_dir, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=args.local_dir,
            token=args.token,
            max_workers=4,
        )
    except GatedRepoError as e:
        print(
            "Access denied or not yet approved for this gated dataset.\n"
            "1) Open https://huggingface.co/datasets/Building3D/Building3D\n"
            "2) Log in, accept the CC BY-NC-SA 4.0 terms, and wait for approval if required.\n"
            "3) Ensure HF_TOKEN is set to a token with read access.\n",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        sys.exit(2)
    print(path)


if __name__ == "__main__":
    main()
