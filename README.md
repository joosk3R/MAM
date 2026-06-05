# MAM: Metal Artifact-Aware Tooth Segmentation

Official implementation of the MICCAI 2026 paper:

**Robust Tooth Segmentation Under Orthodontic CBCT: A Metal Artifact-Aware Approach**  
SeungKwan Jo, Jeonglok Park, So Hyun Kim, and Minyoung Chung

This repository provides the inference code for a metal artifact-aware two-stage framework for robust tooth detection and instance segmentation in orthodontic CBCT.

## Overview

Tooth instance segmentation in Cone-Beam Computed Tomography (CBCT) is important for clinical diagnosis, treatment planning, and prognosis prediction. However, orthodontic CBCT often contains severe metal artifacts caused by brackets or other orthodontic appliances. These artifacts corrupt structural cues, make tooth localization unstable, and degrade final instance segmentation quality.

MAM addresses this problem with a two-stage detection-segmentation pipeline:

1. VOI localization from CBCT DICOM slices.
2. Metal artifact-aware tooth bounding box detection.
3. Patch-based 3D tooth instance segmentation.
4. Export of individual or full-tooth segmentation masks as NIfTI files.

## Repository Structure

```text
MAM/
├── main_test.py                                  # Main inference entry point
├── config.py                                     # Global configuration
├── import_py.py                                  # Shared imports and device setting
├── function_py.py                                # Inference and preprocessing functions
├── utils.py                                      # Utility functions
├── VGG.py                                        # VOI localization network
├── unet.py                                       # 3D U-Net segmentation network
├── model/
│   ├── detection_and_metal_classification_model.py
│   └── resnet.py
├── dataset/
│   └── dataset.py
├── dataloader/
│   └── detection_and_metal_classification_dataset.py
├── pt_file/
│   └── all_model_pt/
│       └── checkpoint/
│           └── README.md
├── examples/
│   └── README.md
├── scripts/
│   └── check_environment.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation

We recommend using a clean conda environment.

```bash
conda create -n mam python=3.9 -y
conda activate mam
pip install -r requirements.txt
```

This code was originally developed for GPU inference. CPU execution may be slow for 3D CBCT volumes.

## Model Weights

Pretrained weights are not included in this repository because model checkpoints can be large.

Place the pretrained checkpoints in the following directory:

```text
pt_file/all_model_pt/checkpoint/
├── model_voi_crop.pth
├── model_bbox.pth
└── model_seg.pt
```

You can also specify a different checkpoint root with `--pt_file` and `--experiment_id`.

## Data Preparation

The input should be a directory containing CBCT DICOM slices for one case.

Example:

```text
sample_case/
├── slice_001.dcm
├── slice_002.dcm
├── ...
└── slice_N.dcm
```

Clinical DICOM/NIfTI data are not included due to privacy restrictions.

## Inference

### Individual tooth segmentation

```bash
python main_test.py \
  --choice 1 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder
```

This saves each predicted tooth mask as a separate `.nii.gz` file.

### Full-tooth segmentation

```bash
python main_test.py \
  --choice 2 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder
```

This saves upper and lower full-tooth segmentation masks as `.nii.gz` files.

### Excluding missing/extracted teeth

If a tooth should be excluded during full-tooth segmentation, pass FDI tooth numbers as a comma-separated list.

```bash
python main_test.py \
  --choice 2 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder \
  --exclude_list 18,28
```

### Custom checkpoint location

```bash
python main_test.py \
  --choice 2 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder \
  --pt_file /path/to/pt_file \
  --experiment_id all_model_pt
```

## Environment Check

```bash
python scripts/check_environment.py
```

## Notes

- This public release focuses on inference.
- Patient data and model checkpoints are intentionally excluded from the repository.
- DICOM, NIfTI, and checkpoint files are ignored by `.gitignore` to prevent accidental upload of clinical data or large binary files.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{jo2026robust,
  title     = {Robust Tooth Segmentation Under Orthodontic CBCT: A Metal Artifact-Aware Approach},
  author    = {Jo, SeungKwan and Park, Jeonglok and Kim, So Hyun and Chung, Minyoung},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```

## Contact

For questions, please open an issue in this repository.
