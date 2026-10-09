# SCALAR-UDA
# SCALAR: Scale Anchoring and Label-shift-Aware Recalibration for Cross-Sensor Landslide Mapping

> **Code Availability:** The source code will be released after the paper is published and indexed.
>
> 本项目代码将在论文正式发表并被检索后开源。

SCALAR is an unsupervised domain adaptation (UDA) framework for cross-sensor landslide segmentation between UAV and satellite imagery.

The repository provides implementations for source-domain training, target-domain adaptation, evaluation, and reproducibility experiments.

## Repository Structure

```text
cas/
├── train.py          # Source-domain training
├── train_uda.py      # Domain adaptation
├── splits.py         # Dataset split construction
└── dump_tiles.py     # Prediction export

splits/
├── index.csv         # Dataset tile index
└── *.json            # Dataset split configurations

scripts/              # Experiment scripts
requirements.lock.txt # Python dependencies
```

## Installation

The implementation was tested with the following environment:

- Python 3.12
- PyTorch 2.8.0
- CUDA 12.8
- segmentation-models-pytorch 0.5.0
- timm 1.0.29
- NVIDIA RTX 5090 (single GPU)
- bf16 mixed precision

Install PyTorch:

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the required dependencies:

```bash
pip install segmentation-models-pytorch==0.5.0 \
  timm==1.0.29 opencv-python tifffile numpy pillow
```

The complete experimental environment is recorded in `requirements.lock.txt`.

## Dataset Preparation

Experiments use the [CAS Landslide Dataset](https://zenodo.org/records/10294997).

1. Download the dataset from the official source.
2. Extract the downloaded subsets into `data/CAS_raw/`.
3. Preserve the original directory structure.
4. Use the provided `splits/index.csv` and JSON split files.

Expected data directory:

```text
data/
└── CAS_raw/
    └── ...           # Original dataset directories

splits/
├── index.csv
├── C_uav2sat.json
└── C_sat2uav.json
```

The dataset is not redistributed in this repository. Users must comply with the original dataset and subset licences.

## Training and Evaluation

The pipeline consists of two stages:

1. Source-domain training with GSD anchoring.
2. Unsupervised domain adaptation initialized from the source checkpoint.

The following commands demonstrate the UAV-to-satellite (UAV→SAT) configuration.

### Step 1. Source-Domain Training

```bash
python -m cas.train \
  --split splits/C_uav2sat.json \
  --index splits/index.csv \
  --raw data/CAS_raw \
  --runs runs \
  --name uav2sat_gsa_s0 \
  --arch segformer \
  --encoder mit_b2 \
  --epochs 30 \
  --bs 16 \
  --lr 6e-5 \
  --seed 0 \
  --gsd_target 1.0 \
  --gsd_clamp 0.25,2.0
```

The source checkpoint is saved under the corresponding experiment directory.

### Step 2. Domain Adaptation

```bash
python -m cas.train_uda \
  --split splits/C_uav2sat.json \
  --index splits/index.csv \
  --raw data/CAS_raw \
  --runs runs \
  --name uav2sat_scalar_s0 \
  --arch segformer \
  --encoder mit_b2 \
  --init runs/uav2sat_gsa_s0/best.pt \
  --epochs 10 \
  --bs 8 \
  --lr 3e-5 \
  --seed 0 \
  --gsd_target 1.0 \
  --gsd_clamp 0.25,2.0 \
  --uda_mode transductive \
  --pl_mode otsu \
  --pl_gate margin \
  --pl_margin 1.0 \
  --pl_margin_pos 1.5 \
  --pl_margin_neg 0.5 \
  --prior_pool 256 \
  --classmix 1 \
  --ema 0.999 \
  --lam_t 1.0
```

### Step 3. Satellite-to-UAV Configuration

For the reverse adaptation direction (SAT→UAV), modify the corresponding arguments in both training stages.

Dataset split:

```bash
--split splits/C_sat2uav.json
```

GSD target:

```bash
--gsd_target 0.5
```

Adaptation margins:

```bash
--pl_margin_pos 0.5 --pl_margin_neg 1.5
```

Use separate experiment names and checkpoint paths for each direction.

## Experiment Outputs

Experiment results are saved under the `runs/` directory.

The selected checkpoint is stored as:

```text
runs/<experiment_name>/best.pt
```

Target-domain evaluation results are written to:

```text
runs/<experiment_name>/metrics_test.json
```

Checkpoints are selected using the source-domain validation split. Target-domain labels are not used for training or checkpoint selection.

## Reproducibility

The repository includes the dataset index, split configurations, and experiment scripts used in the study.

To reproduce the experiments:

- Use the provided dataset splits.
- Follow the specified source-training and adaptation stages.
- Keep the same random seed and training configuration.
- Use the dependency versions specified in `requirements.lock.txt`.

## License

The code license will be specified when the repository is released.

The CAS Landslide dataset is not included in this repository and remains subject to its original licensing terms.

## Acknowledgements

This implementation uses [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) and [timm](https://github.com/huggingface/pytorch-image-models).

We thank the authors of the CAS Landslide dataset for making the dataset publicly available.
