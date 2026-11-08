# SCALAR: Scale Anchoring and Label-Free Recalibration for Cross-Sensor Landslide Mapping

Code for the paper *"SCALAR: Scale Anchoring and Label-Free Recalibration for Cross-Sensor Landslide Mapping between UAV and Satellite Imagery"* (Remote Sensing, under review).

SCALAR adapts a landslide segmentation network from a labelled source platform to an unlabelled target platform (UAV 0.2–1 m ↔ satellite/aerial 0.5–5 m). It adds no trainable parameters to the network. Two operations form the default configuration:

| Component | What it does | Flags |
|---|---|---|
| **GSA** – GSD Anchoring | Every source and target tile is resampled by `clip(gsd_tile / g*, 1/4, 2)` before the network, in training, adaptation and inference. The anchor `g*` is the source GSD closest to the target median GSD: 1.0 m for UAV→SAT, 0.5 m for SAT→UAV. Predictions are restored to the native grid before thresholding. | `--gsd_target g* --gsd_clamp 0.25,2.0` |
| **LOC** – Logit-Otsu Calibration | Inside an EMA-teacher / ClassMix self-training loop, the pseudo-label boundary is the Otsu split of the teacher's target *logit* histogram, re-estimated at every epoch on a fixed pool of 256 unlabelled target tiles. Pixels are retained when their logit lies at least δ = 1 from that boundary (symmetric margins) instead of passing an absolute confidence gate. If no valid two-group split exists, the target pseudo-label loss is skipped for that epoch. | `--pl_mode otsu --pl_gate margin --pl_margin 1.0 --prior_pool 256` |
| **RSR** – Ratio-Signed Regularization (optional) | Makes the two margins asymmetric according to the sign of log r, r = target median GSD / g*: (δ⁺, δ⁻) = (1.5, 0.5) for r > 1 and (0.5, 1.5) for r < 1. Reported in the paper as an exploratory extension; its increments are not statistically significant. | `--pl_margin_pos 1.5 --pl_margin_neg 0.5` (r > 1) or `0.5 / 1.5` (r < 1) |

Checkpoints are selected on the **source** validation split. In the unsupervised adaptation runs no target label enters the adaptation loss, the threshold estimation or the checkpoint selection; target labels are used only to compute the reported metrics. Method design was developed with labelled feedback on the CAS benchmark, as stated in the paper.

## Results (CAS Landslide, SegFormer-B2, landslide IoU at threshold 0.5, five seeds, mean ± sample SD)

| Configuration | UAV → SAT | SAT → UAV |
|---|---|---|
| Source only (native scale) | 0.235 ± 0.014 | 0.289 ± 0.018 |
| Source + GSA | 0.328 ± 0.011 | 0.348 ± 0.008 |
| GSA + LOC | 0.364 ± 0.006 | 0.511 ± 0.017 |
| GSA + LOC + RSR | 0.374 ± 0.008 | 0.527 ± 0.026 |

Region hold-out (Longxi, Moxi and Wenchuan removed from source and adaptation pool, evaluation on the held-out Longxi and Moxi target tiles): GSA + LOC gains 17.9 ± 4.5 IoU points over Source + GSA in SAT→UAV (positive on every seed) and 0.2 ± 6.2 points in UAV→SAT. See the paper for comparators, backbones, external checks (GDCLD, Bijie) and statistics.

## Repository layout

```
cas/                      package (run with python -m cas.<module> from the repository root)
  train.py                source-model training with GSA; also --eval_only inference
  train_uda.py            adaptation: reference loop, LOC, RSR, and the compared UDA methods
  e3_infer.py             one inference pass per checkpoint: operating points, AP, object-level proxies
  build_index.py          tile index (splits/index.csv) from the extracted CAS subsets
  splits.py               protocol splits (A random, B leave-one-region-out, C cross-sensor)
  meta.py                 CAS subset metadata (region, platform, sensor, nominal GSD)
  aug.py, modules.py      augmentation and model helpers used by train.py
  band_table.py           GSD-band aggregation of per-subset metrics
  collect_metrics.py      collects runs/*/metrics_test.json into one table
  make_gdcld_scenes.py    tiles GDCLD scenes (original version); make_external_index.py, eval_external.py
  render_qual.py, dump_tiles.py   qualitative figures
scripts/
  examples.sh             the exact commands of the paper's main configurations
  make_splits_xLMW.py     region-held-out splits (X = Longxi, Moxi, Wenchuan) and the merged LORO fold
  make_gdcld_splits.py  GDCLD scene tiling and the two external splits
  run_e1.py               re-runs a logged argument set with overrides (used for the hold-out experiments)
  bootstrap.py, bootstrap_merge_regions.py, equal_weight*.py, e1_stats.py, e1b_stats.py, e3_table.py,
  e7_logs.py, bijie_stats.py, bijie_ind_half.py   statistics and tables of the paper
  edge_sharpness*.py      relative edge-blur diagnostic (Table A1)
  zenodo_dl.py            resumable multi-connection Zenodo downloader
splits/                   tile index and every split used in the paper (see below)
env/                      frozen package lists of the two machines used
requirements.txt          core dependencies
```

## Installation

Tested with Python 3.12.3 (RTX 5090) and Python 3.11.16 (RTX 4090), PyTorch 2.8.0 + CUDA 12.8, cuDNN 9.10, bfloat16 autocast.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The complete frozen environments are in `env/requirements_rtx5090_py312.lock.txt` and `env/requirements_rtx4090_py311.lock.txt`.

## Data

**CAS Landslide** (Zenodo record 10294997, CC BY-NC 4.0; some subsets carry their own terms, see Table A1 of the paper). The data are not redistributed here.

```bash
python scripts/zenodo_dl.py 10294997 data/CAS          # or download manually
# extract every subset zip under data/CAS_raw/<Subset>/{img,mask}/ as released
python -m cas.build_index --zips data/CAS --raw data/CAS_raw --out splits/index.csv   # optional: splits/index.csv is provided
```

`splits/index.csv` lists every tile (`sid, region, platform, sensor, gsd_m, stem, img, mask, ls_frac, h, w`; paths are relative to `data/CAS_raw`). The split files are JSON with `train`, `val` and `test` lists of `sid/stem` identifiers:

| File | Protocol |
|---|---|
| `A_712.json`, `A_sat_712.json`, `A_uav_712.json`, `A_random73.json` | random within-subset splits (in-domain reference) |
| `B_loro_<Region>.json`, `B_loro_LongxiWenchuan.json` | leave-one-region-out; the merged Longxi + Wenchuan fold replaces the two overlapping folds |
| `C_uav2sat.json`, `C_sat2uav.json` | cross-sensor main protocol (source train/val, target test = adaptation pool) |
| `C_xLMW_<dir>.json`, `C_xLMW_<dir>_holdout.json` | region hold-out: source and pool without Longxi, Moxi, Wenchuan; held-out Longxi + Moxi target tiles |
| `X_bj_sat.json`, `X_bj_uav.json`, `index_bijie.csv` | Bijie external test (CAS source partitions + 2,773 Bijie crops) |
| `PROTOCOL_frozen_20260903.md`, `split_counts*.csv` | protocol description frozen before the main runs, and tile counts |

**GDCLD** (Zenodo 13612636): download the test package, then `python scripts/make_gdcld_splits.py` tiles the scenes into 1024-pixel windows and writes `splits/index_gdcld.csv`, `X_gd_uav2coarse.json` and `X_gd_sat2fine.json` (Palu scenes excluded because the event is in CAS). **Bijie** (Wuhan University GPCV): place the landslide and non-landslide crops under `data/Bijie/`; the index and splits used in the paper are provided in `splits/`. The evaluation plan written before the Bijie download is `splits/bijie_preregistration_20261022.md`.

## Running

All commands are run from the repository root; outputs go to `runs/<name>/` (`log.txt`, `best.pt`, `metrics_val.json`, `metrics_test.json`). `scripts/examples.sh` contains the full argument lists; the essentials are:

**1. Source model with GSA** (30 epochs, batch 16; `--gsd_target 1.0` for UAV→SAT, `0.5` for SAT→UAV)

```bash
python -m cas.train --split splits/C_uav2sat.json --index splits/index.csv --raw data/CAS_raw --runs runs \
  --name C_uav2sat_M1_g1.0_s0 --arch segformer --encoder mit_b2 --epochs 30 --bs 16 --lr 6e-05 --lr_head_mult 10 \
  --wd 0.01 --seed 0 --gsd_target 1.0 --gsd_clamp 0.25,2.0
```

**2. GSA + LOC** (10 epochs, batch 8 + 8, initialised from step 1)

```bash
python -m cas.train_uda --split splits/C_uav2sat.json --index splits/index.csv --raw data/CAS_raw --runs runs \
  --name C_uav2sat_otsuM_s0 --arch segformer --encoder mit_b2 --init runs/C_uav2sat_M1_g1.0_s0/best.pt \
  --epochs 10 --bs 8 --lr 3e-05 --lr_head_mult 10 --wd 0.01 --seed 0 --gsd_target 1.0 --gsd_clamp 0.25,2.0 \
  --uda_mode transductive --pl_mode otsu --pl_gate margin --pl_margin 1.0 --prior_pool 256 \
  --classmix 1 --ema 0.999 --lam_t 1.0 --warm_steps 200
```

**3. + RSR**: add `--pl_margin_pos 1.5 --pl_margin_neg 0.5` for UAV→SAT (r = 3) or `--pl_margin_pos 0.5 --pl_margin_neg 1.5` for SAT→UAV (r = 0.4).

**Other configurations.** Reference loop (DACS-style): `--pl_mode fixed --pl_gate abs --pl_tau 0.9`. Inductive check: `--uda_mode inductive`. Region hold-out: use `splits/C_xLMW_<dir>.json` with the rule anchor `--gsd_target 0.5` in both directions and evaluate the held-out tiles with `python -m cas.train --eval_only runs/<name>/best.pt --split splits/C_xLMW_<dir>_holdout.json ...`. Comparators (`--adv_w/--ent_w`, `--pl_mode cbst`, `--rcs_alpha 0.5`, `--mic_ratio 0.7`, `--wallis 1 --contrast_w 0.5`, `--pl_mode flexmatch|freematch`) and the FDA source training (`cas.train --fam_prob 1.0 --fam_ref target`) are listed in `scripts/examples.sh`. Backbones: `--arch segformer --encoder mit_b2|mit_b0`, `--arch unet|deeplabv3plus --encoder resnet50`, `--arch upernet --encoder tu-convnextv2_tiny|tu-hrnet_w32`.

**Operating-point and object-level analysis** (Table 5 and Appendix C of the paper):

```bash
python -m cas.e3_infer --ckpt runs/C_sat2uav_M1_g0.5_s0/best.pt --split splits/C_sat2uav.json \
  --index splits/index.csv --raw data/CAS_raw --gsd_target 0.5 --out runs/E3/sat2uav_M1_s0
python scripts/e3_table.py          # summary table; scripts/bootstrap*.py, equal_weight*.py, bijie_stats.py for the statistics
```

## Hyper-parameters of the paper

| Item | Value |
|---|---|
| Source training | 30 epochs, batch 16, AdamW, encoder lr 6e-5 (decoder/head ×10), weight decay 0.01, linear warm-up 500 steps then linear decay, BCE + Dice |
| Adaptation | 10 epochs, 8 source + 8 target tiles per step, encoder lr 3e-5 (×10 head), warm-up 200 steps, mixed-view loss disabled for the first 200 steps, EMA 0.999, λ_t = 1, ClassMix on |
| GSA | bounds [1/4, 2]; anchors 1.0 m (UAV→SAT), 0.5 m (SAT→UAV), 0.6 m (CAS SAT→Bijie), 1.0 m (CAS UAV→Bijie), 0.5 m (both directions under region hold-out) |
| LOC | logit histogram of 1000 bins on [−7, 7], pool of 256 target tiles redrawn once per run, threshold clipped to [0.02, 0.98], margin δ = 1 |
| RSR | (1.5δ, 0.5δ) for r > 1, (0.5δ, 1.5δ) for r < 1 |
| Inference | sliding 512 × 512 windows, stride 256, logits restored to the native grid, probability threshold 0.5 |
| Seeds | 0–4 for the main, inductive, ablation and hold-out runs; 0–2 for Bijie; 0 for the diagnostics |

## Notes on this release

- No model weights are distributed; every result in the paper can be regenerated from the commands above on the public datasets.
- `cas/train.py` and `cas/train_uda.py` keep a number of development-stage options (Fourier amplitude mixing, amplitude normalisation, cross-scale consistency, per-tile and soft gates, …) that are **not** part of any reported configuration. Their defaults are off.
- Figure-generation code is not included.
- This archive is the code snapshot of 2026-10-21 that produced the main, ablation, comparator, region-hold-out, GDCLD and operating-point results. Three small additions made for the Bijie runs of 2026-10-22/23 are not yet in it and will be added in the next revision of the archive: the `--tgt_pad_ignore` option of `cas/train_uda.py` (excludes reflection-padded target pixels from the pseudo-label supervision; the Bijie runs used `--tgt_pad_ignore 1`), the edge-padding of inputs to multiples of 32 in `cas/e3_infer.py` (needed for the variable-size Bijie crops), and the script that built `splits/index_bijie.csv` from the downloaded Bijie crops. The inference-timing benchmark script (`bench_infer.py`) is likewise pending.

## Citation

```bibtex
@article{scalar2026,
  title   = {SCALAR: Scale Anchoring and Label-Free Recalibration for Cross-Sensor Landslide Mapping between UAV and Satellite Imagery},
  author  = {[Authors]},
  journal = {Remote Sensing},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The CAS Landslide, GDCLD and Bijie datasets are not part of this repository and remain under their own terms. The implementation builds on [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) and [timm](https://github.com/huggingface/pytorch-image-models).
