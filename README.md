# 3D building construction — CVPR 2026 competition entry

This repo is a workspace for participating in the **CVPR 2026** building / 3D challenges publicized by the organizers (see [announcement](#announcement) below).

## Two live competitions

| Track | What it is | Official hub |
|--------|------------|----------------|
| **BuildingWorld 2026** | 3rd Building3D Challenge — BuildingWorld dataset | [BuildingWorldChallenge on Hugging Face](https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge) |
| **S23DR 2026** | Reconstruct **house roof wireframes** from **point clouds** and **segmentations** | [S23DR2026 on Hugging Face](https://huggingface.co/spaces/usm3d/S23DR2026) |

From the public post: combined prize pool on the order of **$14k USD**, deadline **end of May 2026**. Confirm dates, rules, and prizes on each Space when they are updated.

## How to participate (practical steps)

1. **Create or use a [Hugging Face](https://huggingface.co) account** — Both hubs are HF Spaces; submission and evaluation are typically wired through HF and/or linked platforms described in each Space.
2. **Open each Space in a desktop browser** — The pages host instructions, baselines, submission format, and often a leaderboard. Wait for the Space to finish loading if you see “Fetching metadata…” or a spinner.
3. **Read the rules on each Space** — Team limits, allowed data, citation requirements, and the exact prediction format differ by track.
4. **Get the data** — For S23DR 2026, a public sampled dataset is [`usm3d/s23dr-2026-sampled_4096_v2`](https://huggingface.co/datasets/usm3d/s23dr-2026-sampled_4096_v2) (includes `train` and `validation` splits). BuildingWorld data and download steps are described on the BuildingWorld Space.
5. **Implement and submit** — Build your method in this repo (or elsewhere), produce outputs in the required format, and submit through the channel specified on the corresponding Space.

## Quick links

- BuildingWorld: https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge  
- S23DR 2026: https://huggingface.co/spaces/usm3d/S23DR2026  
- S23DR 2026 dataset (sampled point clouds): https://huggingface.co/datasets/usm3d/s23dr-2026-sampled_4096_v2  

## Announcement

Original LinkedIn post (CVPR 2026 competitions, BuildingWorld + S23DR):  
https://www.linkedin.com/posts/ruisheng-wang-27235a4_cvpr2026-share-7440022939347030017-h1RD
