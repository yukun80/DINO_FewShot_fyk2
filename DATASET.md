# Disaster Dataset Setup

This repository is streamlined for the Exp_Disaster_Few-Shot dataset only.

1) Prepare data directory

- Place the dataset at a local path, e.g. `../_datasets/Exp_Disaster_Few-Shot` with the structure:
```
Exp_Disaster_Few-Shot/
└── valset/
    ├── images/
    │   └── *.tif
    └── labels/
        └── *.tif
```

2) Generate support/query splits

- Use the provided script to generate split JSONs in `datasets/` (paths are saved relative to the project root):
```
python -m datasets.generate_disaster_splits \
  --path ../_datasets/Exp_Disaster_Few-Shot \
  --shots 10 \
  --query 257
```

3) Configure paths

- Edit `configs/disaster.yaml`:
  - `dataset_dir`: absolute or relative path to your dataset directory (e.g., `../_datasets/Exp_Disaster_Few-Shot`).
  - `split_file`: one of the generated JSONs in `datasets/` (e.g., `datasets/disaster_10shot_splits.json`).

Notes
- The dataset class maps label value 20 to class 1 (landslide) and keeps 0 as background.
- All experiments use the split file’s `support` for training and `query` for validation.
