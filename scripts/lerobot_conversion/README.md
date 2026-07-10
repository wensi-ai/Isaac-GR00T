# Converting from LeRobot v3 to v2

## Setup

### 1. Create and Activate Virtual Environment

Run these from the `scripts/lerobot_conversion` directory (it has its own
`pyproject.toml`; installing from the repo root would install the `gr00t`
package instead):
```bash
cd scripts/lerobot_conversion
uv venv
source .venv/bin/activate
uv pip install -e . --verbose
```

### 2. Run Conversion Script

Inside the uv environment, run:
```bash
python convert_v3_to_v2.py --repo-id BobShan/double_folding_towel_v3.0
```

> **Note:** You may need to install lerobot with `GIT_LFS_SKIP_SMUDGE=1`:
> 
> ```bash
> GIT_LFS_SKIP_SMUDGE=1 uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
> ```

## Partial (single-task) subsets of a multi-task dataset

Multi-task v3.0 releases (e.g. BEHAVIOR-1K `2026-challenge-demos`) are often downloaded
per task: the full `meta/**` plus only one task's `data`/`videos` chunks. Episode and
frame indices inside the parquets stay **global** (task 42's episodes start at
`episode_index` 8400 — they do not restart at 0 in a subset).

The converter handles this: episodes listed in `meta/episodes` whose data files are not
present locally are skipped with a warning, and the converted v2.1 layout keeps the global
indices (`data/chunk-008/episode_008400.parquet`, `videos/chunk-008/<key>/episode_008400.mp4`).
`total_episodes` / `total_frames` / `total_chunks` / `total_videos` / `splits` in the
converted `info.json` describe the episodes actually converted. Downstream tooling reading
a converted subset must therefore not assume episode indices start at 0.
