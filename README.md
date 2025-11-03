<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> A Novel Benchmark for Few-Shot Semantic Segmentation in the Era of Foundation Models </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://scholar.google.com/citations?user=InQw64sAAAAJ&hl=fr" target="_blank" style="text-decoration: none;">Reda Bensaid<sup>1,2</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=n3IKEqgAAAAJ&hl=fr" target="_blank" style="text-decoration: none;">Vincent Gripon<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.ca/citations?user=SQrTW_kAAAAJ&hl=en" target="_blank" style="text-decoration: none;">François Leduc-Primeau<sup>2</sup></a>&nbsp;
    <a href="https://scholar.google.com/citations?user=ivJ6Tf8AAAAJ&hl=de" target="_blank" style="text-decoration: none;">Lukas Mauch<sup>3</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.fr/citations?user=FwjpGsgAAAAJ&hl=fr" target="_blank" style="text-decoration: none;">Ghouthi Boukli-Hacene<sup>3,4</sup></a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=UFl8n4gAAAAJ&hl=de" target="_blank" style="text-decoration: none;">Fabien Cardinaux<sup>3</sup></a>&nbsp;,&nbsp;
	<br>
<sup>1</sup>IMT Atlantique&nbsp;&nbsp;&nbsp;
<sup>2</sup>Polytechnique Montréal&nbsp;&nbsp;&nbsp;
<sup>3</sup>Sony Europe, Stuttgart Laboratory 1&nbsp;&nbsp;&nbsp;
<sup>4</sup>MILA&nbsp;&nbsp;&nbsp;

</p>

<p align='center';>

</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2401.11311" target="_blank" style="text-decoration: none;">[Paper]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</b>
</p>


![Alt text](static/main_figure.png)

## Requirements
### Installation
```
python3 -m venv fss_env
source fss_env/bin/activate

pip install -r requirements.txt
```

### Dataset
Follow [DATASET.md](DATASET.md) to prepare the disaster dataset and generate support/query splits.

## Get Started
### Configs
The running configurations can be modified in `configs`. 

### Running with Sacred

With the integration of Sacred, all experiments are now run using a new command structure. This enhances reproducibility and organizes all outputs.

**Base Command Structure:**
`python3 <script_name>.py with <config_key>=<value> ...`

**Example: Training**

The framework now ships with a single decoder path: the multilayer DPT head. The `method` key therefore stays at its default value of `multilayer` for every run.

To train on the `disaster` dataset (20-shot) with DINOv3:
```bash
python3 train.py with dataset=disaster nb_shots=20 dino_version=3 dinov3_size=base run_id=1
```
If your DINOv3 weight filename differs, specify it explicitly:
```bash
python3 train.py with dataset=disaster nb_shots=20 dino_version=3 dinov3_size=base \
  dinov3_weights_path='pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth' run_id=1
```
- To run multiple experiments for averaging, simply increment `run_id` for each run (e.g., `run_id=2`, `run_id=3`).
- All results, logs, and model checkpoints will be saved in a unique directory under `experiments/FSS_Training/`.

**Example: Evaluation**

To evaluate a trained model, you must provide the path to the model checkpoint.
```bash
python3 eval.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
```
- Evaluation metrics will be logged and saved in a new run under `experiments/FSS_Evaluation/`.

Tip: When the checkpoint comes from this repo's Sacred training run, `eval.py` will automatically read the
`config.json` stored next to the checkpoint and reuse backbone settings (e.g., `dino_version`, `dinov2_size`,
`dinov3_size`, `dinov3_weights_path`, `input_size`). This means you can usually omit these flags at eval time.

**Example: Prediction**

To generate segmentation masks from a trained model:
```bash
python3 predict.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
```
- The resulting images will be saved as artifacts in a new run under `experiments/FSS_Prediction/`.

### Visualizing IFA Iterations
To understand how the IFA branch refines predictions over multiple iterations, run:
```bash
python -m tools.visualize_ifa_iterations with \
    checkpoint_path='experiments/FSS_Training/1/best_model.pth' \
    num_samples=4 use_ifa=True ifa_iters=3
```
- Saves side-by-side panels (RGB, GT, decoder, per-iteration IFA, fused) to `experiments/FSS_IFAIterViz/<run_id>/ifa_iterations/`.
- Set `sample_indices=[...]` to inspect fixed queries, `max_iters_to_plot` to clip iterations, and `save_mask_arrays=True` to also export `.npz` files containing the raw masks for each step.

**Available Options:**
- **Scripts**: `train.py`, `eval.py`, `predict.py`
- **Decoder**: fixed multilayer DPT head with evenly spaced layer sampling (e.g., [2,5,8,11] for depth=12) across both DINOv2 and DINOv3.
- **Encoder adapters**: `encoder_adapters ∈ {none, lora, svf}`.
- **Backbones**: `DINO` with `dino_version` in `{2, 3}` and size controls:
  - DINOv2: `dinov2_size ∈ {small, base, large}`
  - DINOv3: `dinov3_size ∈ {small, base, large}`, optional `dinov3_weights_path`
- **Datasets**: The framework is currently optimized for the `disaster` dataset.

### t-SNE Feature Embeddings
For a qualitative look at how query samples spread in feature space, use the dedicated t-SNE visualizer (requires `scikit-learn`):
```bash
python -m tools.visualize_tsne with \
    checkpoint_path='experiments/FSS_Training/1/best_model.pth' \
    feature_stage='decoder_input' sample_mode='pixel' pixels_per_image=1024 max_points=5000 random_seed=0
```
- Supports multiple feature stages (`backbone_raw`, `decoder_input`, `ifa_logits`, etc.) and both image-level or pixel-level sampling.
- The `max_points` cap keeps t-SNE memory manageable; increase cautiously if you have plenty of RAM.
- Saves scatter plots and the underlying embeddings in `experiments/FSS_TSNEViz/<run_id>/tsne_plots/`.

### Frequency Decoupling Modules (FDM)
The framework optionally integrates a lightweight frequency masker (APM) and a channel phase attention (ACPA) between the backbone and decoders. Configure in `configs/disaster.yaml`:

```
fdm:
  enable_apm: false       # APM on/off
  apm_mode: "S"           # "S": [1,1,H,W], "M": [1,C,H,W]
  enable_acpa: false      # ACPA on/off
```

Notes:
- Integrates with the multilayer decoder (and optional SVF/LoRA adapters) while preserving tensor shapes.
- Policy (fixed): the two deepest of the four encoder features receive APM→ACPA before entering the DPT decoder.
- Mixed precision (AMP) is disabled in training to accommodate FFT-based ops (set `mixed_precision: False`).

### FDM Feature Hexbins
To summarize how FDM reshapes backbone activations, generate PCA→HSV hexbin maps:
```bash
python -m tools.fdm_feature_hex \
    --checkpoint_path='experiments/FSS_Training/1/best_model.pth' \
    --stages stage1 stage2 \
    --num_samples 4 \
    --save_projection
```
- Stage 1/2 correspond to pre/post FDM tensors gathered via `forward_with_feature_maps`.
- Each PNG encodes principal-component angle (Hue), energy (Saturation), and third-component brightness (Value).
- When `--save_projection` is set, the JSON summary also reports PCA variance, HSV statistics, and deltas between stages.


## Acknowledgement

This repo benefits from [RegSeg](https://github.com/RolandGao/RegSeg).

## Citation
```latex
@misc{bensaid2024novelbenchmarkfewshotsemantic,
      title={A Novel Benchmark for Few-Shot Semantic Segmentation in the Era of Foundation Models}, 
      author={Reda Bensaid and Vincent Gripon and François Leduc-Primeau and Lukas Mauch and Ghouthi Boukli Hacene and Fabien Cardinaux},
      year={2024},
      eprint={2401.11311},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.11311}, 
}
```

## Contact

If you have any question, feel free to contact reda.bensaid@imt-atlantique.fr.
