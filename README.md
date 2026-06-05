# MAM: Metal Artifact-Aware Tooth Segmentation

Official implementation of:

**Robust Tooth Segmentation Under Orthodontic CBCT: A Metal Artifact-Aware Approach**
SeungKwan Jo, Jeonglok Park, So Hyun Kim, and Minyoung Chung
MICCAI 2026

## Overview

Tooth instance segmentation in Cone-Beam Computed Tomography (CBCT) is essential for clinical diagnosis, treatment planning, and prognosis prediction. However, orthodontic CBCT often contains severe metal artifacts caused by brackets or other orthodontic appliances, which degrade tooth localization and boundary delineation.

This repository provides the implementation of a metal artifact-aware two-stage framework for robust tooth detection and instance segmentation in orthodontic CBCT.

The pipeline consists of:

1. VOI localization from input DICOM files.
2. Metal artifact-aware tooth bounding box detection.
3. Patch-based 3D tooth instance segmentation.
4. Export of individual or full-tooth segmentation results.

## Repository Structure

```text
MAM/
├── main_test.py
├── config.py
├── import_py.py
├── function_py.py
├── utils.py
├── VGG.py
├── unet.py
├── model/
├── dataset/
├── dataloader/
├── test/
└── pt_file/
```

## Installation

We recommend using a clean Python environment.

```bash
conda create -n mam python=3.9
conda activate mam
pip install -r requirements.txt
```

## Model Weights

This code expects pretrained weights in the following directory:

```text
pt_file/all_model_pt/checkpoint/
├── model_voi_crop.pth
├── model_bbox.pth
└── model_seg.pt
```

Due to file size and data-sharing constraints, pretrained weights are not included directly in this repository. Please download the weights from the provided link or place your trained checkpoints in the directory above.

## Data Preparation

The input should be a directory containing CBCT DICOM files.

Example:

```text
sample_case/
├── slice_001.dcm
├── slice_002.dcm
├── ...
└── slice_N.dcm
```

Clinical DICOM/NIfTI data are not included in this repository due to privacy restrictions.

## Inference

For individual tooth segmentation:

```bash
python main_test.py \
  --choice 1 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder
```

For full-tooth segmentation:

```bash
python main_test.py \
  --choice 2 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder
```

To exclude specific tooth indices during full-tooth segmentation:

```bash
python main_test.py \
  --choice 2 \
  --directory /path/to/dicom_folder \
  --save_dir /path/to/output_folder \
  --exclude_list 18,28
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{jo2026robust,
  title     = {Robust Tooth Segmentation Under Orthodontic CBCT: A Metal Artifact-Aware Approach},
  author    = {Jo, SeungKwan and Park, Jeonglok and Kim, So Hyun and Chung, Minyoung},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```

## License

This project is released for research purposes. Please refer to the LICENSE file for details.

## Contact

For questions, please contact the corresponding author or open an issue in this repository.
