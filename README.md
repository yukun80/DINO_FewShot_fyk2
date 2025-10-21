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

To run training for DINOv2 with linear probing on the `disaster` dataset (10-shot):
```bash
python3 train.py with method=linear dataset=disaster nb_shots=10 lr=0.01 run_id=1
```
- To run multiple experiments for averaging, simply increment the `run_id` for each run (e.g., `run_id=2`, `run_id=3`).
- All results, logs, and model checkpoints will be saved in a unique directory under `experiments/FSS_Training/`.

To run training for DINOv3 with multilayer decoding on the `disaster` dataset (20-shot):
```bash
python3 train.py with method=multilayer dataset=disaster nb_shots=20 dino_version=3 dinov3_size=base run_id=1
```
If your DINOv3 weight filename differs, specify it explicitly:
```bash
python3 train.py with method=multilayer dataset=disaster nb_shots=20 dino_version=3 dinov3_size=base \
  dinov3_weights_path='pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth' run_id=1
```

**Example: Evaluation**

To evaluate a trained model, you must provide the path to the model checkpoint.
```bash
python3 eval.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
```
- Evaluation metrics will be logged and saved in a new run under `experiments/FSS_Evaluation/`.

Tip: When the checkpoint comes from this repo's Sacred training run, `eval.py` will automatically read the
`config.json` stored next to the checkpoint and reuse backbone settings (e.g., `dino_version`, `dinov2_size`,
`dinov3_size`, `dinov3_weights_path`, `method`, `input_size`). This means you can usually omit these flags at eval time.

**Example: Prediction**

To generate segmentation masks from a trained model:
```bash
python3 predict.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
```
- The resulting images will be saved as artifacts in a new run under `experiments/FSS_Prediction/`.

**Available Options:**
- **Scripts**: `train.py`, `eval.py`, `predict.py`
- **Methods**: `linear`, `multilayer`, `svf`, `lora`.
- **Backbones**: `DINO` with `dino_version` in `{2, 3}` and size controls:
  - DINOv2: `dinov2_size ∈ {small, base, large}`
  - DINOv3: `dinov3_size ∈ {small, base, large}`, optional `dinov3_weights_path`
- **Datasets**: The framework is currently optimized for the `disaster` dataset.


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
