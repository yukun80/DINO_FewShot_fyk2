# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `train.py`, `eval.py`, `predict.py` (Sacred-managed runs).
- Configs: `configs/` (e.g., `configs/disaster.yaml`).
- Data: `datasets/` (disaster-only dataset class, split JSONs, split generator).
- Models: `models/backbones/` (e.g., `dino.py`) and `models/svf.py`.
- Utilities: `utils/` (losses, schedulers, transforms, dataloaders).
- Pretrained weights: `pretrain/` (DINOv2/DINOv3 checkpoints).
- Experiment outputs: created under `experiments/` by Sacred; keep out of git.
  - Sacred controls run directories; YAML has no `save_dir`.

## Build, Test, and Development Commands
- Setup (recommended venv):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Generate splits (example):
  - `python -m datasets.generate_disaster_splits --path ../_datasets/Exp_Disaster_Few-Shot --shots 10 --query 257`
- Train (example, 10-shot DINOv2 linear):
  - `python3 train.py with method=linear dataset=disaster nb_shots=10 lr=0.01 run_id=1`
- Train (example, 20-shot DINOv3 multilayer):
  - `python3 train.py with method=multilayer dataset=disaster nb_shots=20 dino_version=3 dinov3_size=base run_id=1`
- Evaluate (requires checkpoint):
  - `python3 eval.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10`
- Predict (save masks as artifacts):
  - `python3 predict.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10`
 - Supported `method` values: `linear`, `multilayer`, `svf`, `lora` (VPT removed).

## Coding Style & Naming Conventions
- Python, 4-space indentation, PEP 8. Prefer type hints where helpful.
- Names: functions/vars `lower_snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Config keys use `snake_case`; keep defaults in YAML under `configs/`.
- Place new backbones in `models/backbones/`; dataset logic in `datasets/`; training utilities in `utils/`.
- Only DINOv2 and DINOv3 are supported backbones (DINOv1 removed).

## Testing Guidelines
- No formal unit test suite. Use quick smoke runs:
  - Small split (`--shots 1`) and verify training loop and evaluation complete without errors.
  - Use `eval.py` to confirm metrics log and model loads correctly.
- Keep runs reproducible: set `run_id` (affects RNG seed) and record CLI commands.

## Commit & Pull Request Guidelines
- Commits: imperative, present tense; concise summary line (e.g., "Add DINOv2 multilayer support").
- PRs should include:
  - Purpose and high-level changes; affected files/dirs.
  - Exact commands to reproduce (train/eval/predict) and expected artifact paths under `experiments/`.
  - Notable config changes (e.g., `configs/disaster.yaml`).
  - Avoid committing datasets, checkpoints, and `experiments/` outputs.

## Security & Configuration Tips
- Verify paths in `configs/disaster.yaml` (`dataset_dir`, `model_path`, `model_repo_path`) exist locally.
- Do not commit private data or large weights; `.gitignore` already excludes experiment dirs.
- Some methods (`svf`, `lora`) require a prior linear decoder; ensure `linear_weights_path` is set.
- `model_path` (pretrained weights dir) and `model_repo_path` (local backbone repo) are required for model init.
- For DINOv3, if auto-discovery fails, specify `dinov3_weights_path` explicitly.
- Sacred manages output directories; do not add `save_dir` in YAML. Use `checkpoint_path` for eval/predict.
