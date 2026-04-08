# BuildingWorld 2026 (3rd Building3D)

Official competition Space (registration, leaderboard, instructions):  
https://huggingface.co/spaces/BuildingWorld/BuildingWorldChallenge

The Space runs the standard Hugging Face **Competitions** image (`huggingface/competitions`). Use the UI there for team signup, data access, and submission packaging.

## Building3D dataset (from the reconstruction page)

The [Building3D reconstruction overview](https://szusic.github.io/Building3D/reconstruction.html) links to the Hugging Face dataset **[Building3D/Building3D](https://huggingface.co/datasets/Building3D/Building3D)** (entry-level roof point clouds & wireframe, Tallinn city variants, etc.).

That repository is **gated**: log in on Hugging Face, open the dataset card, **acknowledge the CC BY-NC-SA 4.0 license**, and wait if the maintainers need to approve access (the card may say processing takes a few days).

### Download into this repo

From the repository root (with `huggingface_hub` installed, e.g. `pip install -r requirements.txt`):

1. Create a read token: https://huggingface.co/settings/tokens  
2. Authenticate **one** of these ways:
   - `export HF_TOKEN=hf_...` for scripts, or  
   - `hf auth login` (replacing the old `huggingface-cli login`), or  
   - `python3 -m huggingface_hub.cli.hf auth login` if `hf` is not on your `PATH` (often fix with `export PATH="$HOME/.local/bin:$PATH"` after `pip install --user`).  
3. Run:

```bash
python3 buildingworld/scripts/download_building3d.py --local-dir outputs/building3d_hf
```

Files are written under `outputs/` (gitignored). Use `--repo Building3D/Building3D` to override the default dataset id.

## In this repo

Put BuildingWorld-specific code, configs, and experiment notes here (e.g. `src/`, `configs/`). Keep outputs and downloaded challenge data **out of git** (see root `.gitignore`).

Until the organizers publish baselines or starter kits elsewhere, treat the Space as the source of truth for task definition and file formats.
