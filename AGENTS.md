# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `train.py`, `eval.py`, `predict.py` (Sacred-managed runs).
- Configs: `configs/` (e.g., `configs/disaster.yaml`).
- Data: `datasets/` (disaster-only dataset class, split JSONs, split generator).
- Models: `models/backbones/` (e.g., `dino.py`), `models/decoders/` (e.g., `dpt.py`), and `models/svf.py`.
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

## Frequency Decoupling Modules (FDM)
- Purpose: Optional target-domain finetuning adapters that operate in frequency/phase space to reduce inter-channel correlation and improve cross-domain robustness.
- Modules: APM (Amplitude-Phase Masker) and ACPA (Adaptive Channel Phase Attention).
- Enable in `configs/disaster.yaml` under `fdm`:
  - `enable_apm: {true|false}`
  - `apm_mode: {"S"|"M"}` — "S" uses `[1,1,H,W]` masks; "M" uses `[1,C,H,W]` masks.
  - `enable_acpa: {true|false}`
- Fixed integration policy (no extra knobs):
  - `linear` and `svf`: apply APM → ACPA on the final feature map, then BN + 1×1.
  - `multilayer`: apply APM → ACPA on only the deeper two of the four intermediate features; pass all four into DPT.
- Implementation notes:
  - Batch‑agnostic APM parameters: masks are initialized lazily to `[1,1,H,W]` or `[1,C,H,W]` at first use.
  - Shapes are preserved; decoders (linear/DPT) require no changes.
  - Code points: see `models/backbones/dino.py` around the decoder paths.

Notes on methods
- `multilayer`: Uses a SegDINO-aligned DPT decoder with spread layer sampling.
  - Layer selection: 4 evenly spaced intermediate layers (e.g., depth=12 → [2,5,8,11]).
  - Decoder: per-layer 1x1 projection → 3x3 refinement to `features` with BN+GELU → concat → final 1x1.
  - Applies to both DINOv2 and DINOv3 backbones; backbone forward is computed under no_grad.

## DINOv3 Adaptation
- Robust checkpoint loading with key normalization and validation.
  - Strips common prefixes (`module.`, `backbone.`, `model.`, `teacher.`, `student.`) and unwraps (`state_dict`, `model`, `teacher`, `student`).
  - Loads only matching tensors (key and shape), logs loaded/skipped/missing counts.
  - Fails fast if load ratio is too low (<85%) to avoid silent degradation.
- Auto-detects and configures architectural options from the checkpoint:
  - `n_storage_tokens` if `storage_tokens` present in the weights.
  - LayerScale if `*.ls1.gamma` / `*.ls2.gamma` found.
  - Masked K-bias in attention if `*.attn.qkv.bias_mask` found.
- RoPE dtype configurable via `dinov3_rope_dtype` (default `bf16`), can be set to `fp32` or `fp16` per environment.

## Data Preprocessing
- Disaster dataset images are normalized to ImageNet convention in `datasets/disaster.py`:
  - If values appear in 0–255, they are scaled to [0,1] and clamped.
  - Standardized with mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
- If adding a new dataset class, reproduce the same normalization logic or extract a reusable transform.

Training Stability Tips
- Recommended starting hyperparameters for `multilayer`:
  - DINOv2: `lr≈5e-4`, `weight_decay≈1e-4`, `momentum=0.9`.
  - DINOv3: `lr≈1e-3`, `weight_decay≈1e-4`, `momentum=0.9`.
- Optional gradient clipping: set `clip_grad_norm` (e.g., `clip_grad_norm: 1.0`).
- Mixed precision is disabled; training runs in FP32 to accommodate FFT/phase ops (`mixed_precision: False`).
- Optimizer config values are parsed robustly, but prefer numeric (non-quoted) values for `lr`, `momentum`, `weight_decay`.

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
- For DINOv3, you may also set `dinov3_rope_dtype` (default `bf16`) to `{bf16, fp32, fp16}`.
- Sacred manages output directories; do not add `save_dir` in YAML. Use `checkpoint_path` for eval/predict.
