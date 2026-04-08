# 3D building construction — CVPR 2026 (both tracks)

This repo is a workspace for **both** CVPR 2026 challenges announced in the [LinkedIn post](#announcement):

| Track | Role in this repo | Official hub |
|--------|-------------------|--------------|
| **BuildingWorld 2026** (3rd Building3D) | Code and notes in [`buildingworld/`](buildingworld/) | [BuildingWorldChallenge](https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge) |
| **S23DR 2026** (roof wireframes from point clouds + segmentations) | Code and notes in [`s23dr/`](s23dr/) | [S23DR2026](https://huggingface.co/spaces/usm3d/S23DR2026) |

## Shared setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Hugging Face**: create an account and sign in (`huggingface-cli login`) if a dataset or submission flow requires it.
- **Outputs / downloads**: keep large artifacts under `outputs/` or outside the repo; they are gitignored.

## Participating in both competitions

1. Open each **Space** in a browser and complete **registration / team** steps there (both Spaces use the Hugging Face **Competitions** stack).
2. Read **task definition, baselines, and submission format** on each Space; they are independent.
3. **S23DR** published rules: [Rules.md on the Space](https://huggingface.co/spaces/usm3d/S23DR2026/raw/main/Rules.md) (eligibility, prizes, data license, submission via the Space).
4. **Data**
   - S23DR sampled set: [`usm3d/s23dr-2026-sampled_4096_v2`](https://huggingface.co/datasets/usm3d/s23dr-2026-sampled_4096_v2) — optional peek: `python s23dr/scripts/inspect_dataset.py`
   - Building3D (linked from the [reconstruction page](https://szusic.github.io/Building3D/reconstruction.html)): gated HF dataset [`Building3D/Building3D`](https://huggingface.co/datasets/Building3D/Building3D) — after access is granted, run `python3 buildingworld/scripts/download_building3d.py` (see [`buildingworld/README.md`](buildingworld/README.md)).
5. **Submit** each entry through the corresponding Space; do not assume one submission covers both tracks.

## Quick links

- BuildingWorld: https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge  
- S23DR 2026: https://huggingface.co/spaces/usm3d/S23DR2026  
- S23DR rules: https://huggingface.co/spaces/usm3d/S23DR2026/raw/main/Rules.md  
- S23DR dataset: https://huggingface.co/datasets/usm3d/s23dr-2026-sampled_4096_v2  

## Announcement

Original LinkedIn post:  
https://www.linkedin.com/posts/ruisheng-wang-27235a4_cvpr2026-share-7440022939347030017-h1RD
