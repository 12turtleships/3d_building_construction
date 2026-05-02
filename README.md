# 3D building construction — CVPR 2026 (both tracks)

This repo is a workspace for **both** CVPR 2026 challenges:

| Track | Folder | Official hub | Status |
|---|---|---|---|
| **S23DR 2026** — roof wireframes from point clouds | [`s23dr/`](s23dr/) | [S23DR2026](https://huggingface.co/spaces/usm3d/S23DR2026) | Active |
| **BuildingWorld 2026** — 3rd Building3D | [`buildingworld/`](buildingworld/) | [BuildingWorldChallenge](https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge) | Not started |

---

## S23DR 2026

**Task**: given a 4096-point normalised roof point cloud, predict the roof wireframe graph — a set of 3D vertices and typed edges (ridge, hip, valley, eave, gable).  
**Dataset**: `usm3d/s23dr-2026-sampled_4096_v2` — 15,892 train / 1,024 val samples.  
**Prize**: $5K / $3K / $2K (must beat organiser baseline).

### Approach — CSG-based roof wireframe extraction

We model roofs as **Constructive Solid Geometry (CSG) trees** of typed primitives. A neural network predicts vertex positions and edge connectivity directly; the CSG module provides a geometric prior for post-processing and structured regularisation.

```
Point cloud (1024 pts, xyz + vote_frac + class_id + …)
        ↓
PointNet backbone  →  global descriptor (1024-d)
        ↓
Vertex decoder     →  64 query slots → predicted vertex positions + confidence
        ↓
Edge predictor     →  pairwise → edge class logits (10 classes + "no edge")
        ↓
Hungarian loss     →  match predictions to GT wireframe, train end-to-end
```

#### CSG roof primitives (`s23dr/csg/`)

| Primitive | Shape | CSG role |
|---|---|---|
| `RidgePrism` | Wedge along ridge line | Gable and hip roof sections |
| `HipEnd` | 4-sided pyramid | Hip end caps — `Union` with RidgePrism |
| `FlatSlab` | Rectangular box | Flat roof, parapet, terrace |
| `ValleyPrism` | Inverted wedge | Valley gutter — `Difference` |
| `DormerBox` | Vertical box | Dormer, chimney — `Union` |

Boolean ops: **Union** (cross-gable → valley seam), **Intersection** (clip), **Difference** (carve dormer opening).  
`evaluate_csg(tree)` converts a CSG tree → `(vertices, edges, edge_classes)` matching the competition's GT format.

#### Model (`s23dr/model/`)

| File | Purpose |
|---|---|
| `backbone.py` | PointNet encoder — shared MLP + global max-pool |
| `decoder.py` | `VertexDecoder` (K=64 queries), `EdgePredictor` (pairwise MLP) |
| `loss.py` | Hungarian vertex matching + edge cross-entropy; `decode()` for inference |
| `net.py` | `RoofWireframeNet` — 1.94 M parameters |

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch scipy
```

### Inspect the dataset

```bash
# Metadata peek (one row)
python s23dr/scripts/inspect_dataset.py --split train

# Full payload structure (ZIP arrays + shapes)
python s23dr/scripts/inspect_payload.py --split train --streaming
```

### Run tests

```bash
python -m pytest s23dr/csg/tests/ -v   # 24/24 CSG geometry tests
```

### Train locally (CPU, debug)

```bash
python -m s23dr.train --split-debug --epochs 5 --n-points 1024 --batch-size 2
```

### Train on GPU (Colab — recommended)

Open [`s23dr/colab_train.ipynb`](s23dr/colab_train.ipynb) in Google Colab:

> **File → Open notebook → GitHub → `12turtleships/3d_building_construction` → `s23dr/colab_train.ipynb`**

Set runtime to **T4 GPU** (free) or **A100** (Colab Pro).

| Hardware | Time / epoch | 100 epochs |
|---|---|---|
| T4 GPU (free Colab) | ~3.6 min | ~6 hrs |
| A100 (Colab Pro) | ~1.2 min | ~2 hrs |
| Mac CPU | ~53 min | not viable |

---

## Shared setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Outputs and large artifacts live under `outputs/` (gitignored).

---

## Quick links

- S23DR 2026 Space: https://huggingface.co/spaces/usm3d/S23DR2026
- S23DR dataset: https://huggingface.co/datasets/usm3d/s23dr-2026-sampled_4096_v2
- S23DR rules: https://huggingface.co/spaces/usm3d/S23DR2026/raw/main/Rules.md
- BuildingWorld Space: https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge
- Announcement: https://www.linkedin.com/posts/ruisheng-wang-27235a4_cvpr2026-share-7440022939347030017-h1RD
