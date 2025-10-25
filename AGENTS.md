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
 - IFA modules:
   - `modules/module_IFA/ifa_head.py`: channel‑agnostic, multi‑iteration IFA/BFP/SSP head（训练与推理均可调用）。
   - `utils/ifa.py`: 编码器特征提取、可选的 FDM-on-features、公用 IFA 入口（`run_ifa_inference` 与 `run_ifa_training_logits`）。

## Build, Test, and Development Commands
- Setup (recommended venv):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Generate splits (example):
  - `python -m datasets.generate_disaster_splits --path ../_datasets/Exp_Disaster_Few-Shot --shots 10 --query 257`
- Train（YAML 驱动的最简命令）:
  - `python3 train.py with run_id=1`
- Train（需要时从 CLI 覆盖关键项）:
  - `python3 train.py with method=multilayer encoder_adapters=lora use_ifa_train=True ifa_alpha=0.3 run_id=1`
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

## IFA Integration（Training + Inference）
- Purpose: Optional, training‑free enhancement that iteratively refines foreground/background prototypes (BFP/SSP) on encoder features to improve cross‑domain few‑shot segmentation.
- Inference: Integrated into `predict.py` and `eval.py`. Enabled via Sacred keys.
- Training: 可选并行分支，融合监督：CE(main) + λ_main·CE(fused) + λ_aux·CE(ifa)。
- Support set for IFA:
  - Uses the training split (`train_split` / “support”) from your split file.
  - K 取自 `number_of_shots`。训练可用 `ifa_train_support_k` 限制每迭代参与的支持数（≤K）。
- Multi‑iteration + refine:
  - `ifa_iters` controls the number of BFP iterations (default 3).
  - `ifa_refine=True` enables an extra refine step in the first iteration (as in the original IFA).
- Linear vs. Multilayer:
  - `linear`: IFA runs on the last encoder feature and its logits are upsampled and fused.
  - `multilayer`: IFA runs on the four spread layers; per‑scale logits are upsampled and fused via `ifa_ms_weights` (default `[0.1,0.2,0.3,0.4]`, deeper layers higher weight).
- FDM parity on features:
  - Set `ifa_use_fdm=True` (default) to apply APM→ACPA to the encoder features used by IFA, with the same policy as training: deeper‑two for `multilayer`, last‑only for `linear`.
  - FDM is applied only if the model actually has APM/ACPA enabled; otherwise it is skipped.
- Logit fusion with the trained decoder output:
  - Final logits = `(1‑ifa_alpha) * base_logits + ifa_alpha * ifa_logits` (default `ifa_alpha=0.3`).
- Hyper‑parameters（YAML/CLI）:
  - Inference: `use_ifa`, `ifa_iters`, `ifa_refine`, `ifa_alpha`, `ifa_ms_weights`, `ifa_temp`, `ifa_fg_thresh`, `ifa_bg_thresh`, `ifa_use_fdm`。
  - Training: `use_ifa_train`, `ifa_loss_w_main`, `ifa_loss_w_aux`, `ifa_train_support_k`（参与支持数），`ifa_detach_support`（支持特征 no_grad 抽取）。

### Quick Commands
- Predict with IFA (multilayer example):
  - ``python3 predict.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' method=multilayer nb_shots=10 use_ifa=True ifa_use_fdm=True ifa_iters=3 ifa_refine=True``
- Evaluate with IFA and report deltas (Base vs. IFA):
  - ``python3 eval.py with checkpoint_path='experiments/FSS_Training/dinov2_multilayer+fdm/best_model.pth' nb_shots=20 use_ifa=True ifa_use_fdm=True ifa_iters=3 ifa_refine=True``
  - Eval logs both `Base_*` and `IFA_*` metrics and `Delta_*` (e.g., `Delta_mIoU`).

Notes on methods
- `multilayer`: Uses a SegDINO-aligned DPT decoder with spread layer sampling.
  - Layer selection: 4 evenly spaced intermediate layers (e.g., depth=12 → [2,5,8,11]).
  - Decoder: per-layer 1x1 projection → 3x3 refinement to `features` with BN+GELU → concat → final 1x1.
  - Applies to both DINOv2 and DINOv3 backbones；是否 no_grad 由 `encoder_adapters` 决定：`none`→冻结（no_grad），`lora/svf`→可微（仅训练轻量适配层）。

## Encoder Adapters（解耦）
- `encoder_adapters: {none|lora|svf}` 与解码器类型（`method: linear|multilayer`）解耦。
- 选择 `lora/svf` 时在 encoder 注入轻量适配器、允许可微前向；选择 `none` 则保持冻结、节省显存。

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

## Memory Tuning
- Reduce IFA participation:
  - `ifa_train_support_k`（训练参与的支持数）调低例如 5/10；`ifa_detach_support: true`（默认）以 no_grad 抽取支持特征。
  - `ifa_iters: 2` 或 `ifa_refine: false`；将 `ifa_ms_weights` 浅层置零如 `[0,0,0,1]`。
- Lower FDM/encoder cost:
  - `fdm.apm_mode: "S"` 或临时关闭 `fdm.enable_acpa`。
  - 使用更小 `input_size` 或更小 DINO 尺寸（`dinov2_size: small`/`dinov3_size: small`）。

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
 - For IFA eval/predict smoke‑tests:
   - Start with `nb_shots=1` and `use_ifa=True ifa_iters=3 ifa_alpha=0.3`.
   - For multilayer models trained with FDM, keep `ifa_use_fdm=True` to mirror training features.

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
 - Sacred 使用追踪：新增嵌套键（如 `fdm.apm_mode`）需在捕获函数内显式访问；本工程已在 train/eval/predict 中触碰相关键，避免 ConfigAddedError。
