"""Where the benchmark scripts find things, without hardcoding one machine's layout.

These scripts each carried an absolute path -- `/workspace/DDPMLibrary` on the GPU
box, `/Users/henryw/...` locally -- so a script written on one machine would not run
on the other, and none of them ran for a collaborator or a reviewer.

Repo paths now derive from this file's location. Anything genuinely outside the repo
comes from an environment variable whose default is the original path, so existing
invocations keep working unchanged:

    DDPM_MODELS_DIR    training checkpoints kept outside the repo
    DDPM_PICKLES_DIR   the dataset pickles
    DDPM_SCRATCH_DIR   GPU-box training output (default /workspace)

Each defaults to the layout next to this repo, then to the original absolute path,
so existing invocations keep working.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_sibling = ROOT.parent / "DiffusionSummer2026" / "Models"
MODELS_DIR = Path(os.environ.get(
    "DDPM_MODELS_DIR",
    _sibling if _sibling.exists()
    else "/Users/henryw/Documents/DiffusionSummer2026/Models"))


_pk = ROOT.parent / "DiffusionSummer2026" / "Datasets" / "pickles"
PICKLES_DIR = Path(os.environ.get(
    "DDPM_PICKLES_DIR",
    _pk if _pk.exists()
    else "/Users/henryw/Documents/DiffusionSummer2026/Datasets/pickles"))

#: Where a GPU box wrote training runs. Only used by scripts that score those runs.
SCRATCH_DIR = Path(os.environ.get("DDPM_SCRATCH_DIR", "/workspace"))

