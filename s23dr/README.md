# S23DR 2026 (roof wireframes from point clouds)

Official competition Space (submission + evaluation flow):  
https://huggingface.co/spaces/usm3d/S23DR2026

Published rules (eligibility, prizes, data license):  
https://huggingface.co/spaces/usm3d/S23DR2026/raw/main/Rules.md

Public sampled dataset (train / validation, binary payloads per row):  
https://huggingface.co/datasets/usm3d/s23dr-2026-sampled_4096_v2

## Prizes (from published rules)

- 1st: $5,000 · 2nd: $3,000 · 3rd: $2,000 · additional pool: $2,000  
- Monetary awards apply only to submissions that **beat the published baseline** (see Space).

## In this repo

- `scripts/inspect_dataset.py` — optional helper to stream one example and print field shapes (requires network + HF token if the dataset is gated).

Implement training and inference under this folder; align outputs with the format required on the submission Space.
