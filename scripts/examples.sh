#!/usr/bin/env bash
# Exact configurations used in the paper (run from the repository root; seed 0 shown, seeds 0-4 were run).
# Replace the split/anchor pair for the other direction: C_uav2sat.json with --gsd_target 1.0, C_sat2uav.json with 0.5.
set -e
IDX=splits/index.csv; RAW=data/CAS_raw; RUNS=runs
COMMON_SRC="--arch segformer --encoder mit_b2 --epochs 30 --bs 16 --lr 6e-05 --lr_head_mult 10.0 --wd 0.01 --workers 12 --eval_stride 256 --val_every 1"
COMMON_UDA="--arch segformer --encoder mit_b2 --epochs 10 --bs 8 --lr 3e-05 --lr_head_mult 10.0 --wd 0.01 --workers 10 --uda_mode transductive --pl_tau 0.9 --pl_thresh 0.5 --tgt_scale 1.0 --eval_bs 16 --prior_pool 256 --lam_t 1.0 --ema 0.999 --classmix 1 --warm_steps 200 --eval_stride 256"

# --- Source models ---------------------------------------------------------------------------------------------
# native scale (no GSA)
python -m cas.train --split splits/C_sat2uav.json --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_segformer_b2_s0 $COMMON_SRC --seed 0
# Source + GSA (anchor 0.5 m for SAT->UAV, 1.0 m for UAV->SAT)
python -m cas.train --split splits/C_sat2uav.json --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_M1_g0.5_s0 $COMMON_SRC --seed 0 --gsd_target 0.5 --gsd_clamp 0.25,2.0
python -m cas.train --split splits/C_uav2sat.json --index $IDX --raw $RAW --runs $RUNS --name C_uav2sat_M1_g1.0_s0 $COMMON_SRC --seed 0 --gsd_target 1.0 --gsd_clamp 0.25,2.0
# FDA comparator (source-supervised, Fourier amplitude replacement with target references)
python -m cas.train --split splits/C_sat2uav.json --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_FDA_M1_s0 $COMMON_SRC --seed 0 --gsd_target 0.5 --gsd_clamp 0.25,2.0 --fam_prob 1.0 --fam_beta 0.01,0.09 --fam_alpha 1.0 --fam_ref target

# --- Adaptation, SAT->UAV (r = 0.4) ----------------------------------------------------------------------------
INIT=runs/C_sat2uav_M1_g0.5_s0/best.pt; SPL=splits/C_sat2uav.json; G=0.5
# GSA + LOC (default SCALAR configuration)
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_M1ST_otsuM_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0
# GSA + LOC + RSR (r < 1: positive margin 0.5 delta, negative margin 1.5 delta)
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_M1ST_asym_p05n15_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0 --pl_margin_pos 0.5 --pl_margin_neg 1.5
# reference loop = DACS-style (fixed 0.5 boundary, absolute confidence gate 0.9)
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_M1ST_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode fixed --pl_gate abs
# comparators on the same anchored initialisation
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_advent_M1_s0   --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode fixed --pl_gate abs --adv_w 0.001 --ent_w 0.001 --adv_lr 0.0001 --lam_t 0.0
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_cbst_M1_s0     --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode cbst --pl_gate abs --cbst_p0 0.2 --cbst_dp 0.05
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_daformer_M1_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode fixed --pl_gate abs --rcs_alpha 0.5
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_mic_M1_s0      --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode fixed --pl_gate abs --mic_ratio 0.7 --mic_w 1.0 --mic_patch 64
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_landsda_M1_s0  --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode fixed --pl_gate abs --adv_w 0.001 --wallis 1 --contrast_w 0.5 --contrast_tau 0.1 --lam_t 0.0
# adaptive-gate controls (FlexMatch-/FreeMatch-style gates in the reference loop)
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_flex_M1_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode flexmatch --pl_gate abs
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_free_M1_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode freematch --pl_gate abs
# inductive variant (per-subset 50/50 adaptation / test halves)
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_sat2uav_M1ST_otsuM_ind_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --uda_mode inductive --pl_mode otsu --pl_gate margin --pl_margin 1.0

# --- Adaptation, UAV->SAT (r = 3): same commands with the other split / anchor / RSR sides ------------------------
INIT=runs/C_uav2sat_M1_g1.0_s0/best.pt; SPL=splits/C_uav2sat.json; G=1.0
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_uav2sat_M1ST_otsuM_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0
python -m cas.train_uda --split $SPL --index $IDX --raw $RAW --runs $RUNS --name C_uav2sat_M1ST_asym_p15n05_s0 --init $INIT --seed 0 --gsd_target $G --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0 --pl_margin_pos 1.5 --pl_margin_neg 0.5

# --- Region hold-out (Longxi, Moxi, Wenchuan removed; rule anchor 0.5 m in both directions) ----------------------
python scripts/make_splits_xLMW.py --index $IDX --out splits        # writes C_xLMW_*.json and B_loro_LongxiWenchuan.json
python -m cas.train     --split splits/C_xLMW_sat2uav.json --index $IDX --raw $RAW --runs $RUNS --name C_xLMW_sat2uav_M1_g0.5_s0 $COMMON_SRC --seed 0 --gsd_target 0.5 --gsd_clamp 0.25,2.0
python -m cas.train_uda --split splits/C_xLMW_sat2uav.json --index $IDX --raw $RAW --runs $RUNS --name C_xLMW_sat2uav_otsuM_s0 --init runs/C_xLMW_sat2uav_M1_g0.5_s0/best.pt --seed 0 --gsd_target 0.5 --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0
# held-out Longxi + Moxi tiles (never in the source or the pool): evaluate the selected checkpoints
python -m cas.train --eval_only runs/C_xLMW_sat2uav_otsuM_s0/best.pt --split splits/C_xLMW_sat2uav_holdout.json --index $IDX --raw $RAW --runs $RUNS --name C_xLMW_sat2uav_otsuM_s0_holdout --arch segformer --encoder mit_b2 --gsd_target 0.5 --gsd_clamp 0.25,2.0

# --- External checks -------------------------------------------------------------------------------------------
# GDCLD: tile the test scenes and build splits/index_gdcld.csv, X_gd_sat2fine.json, X_gd_uav2coarse.json
python scripts/make_gdcld_splits.py
python -m cas.train_uda --split splits/X_gd_sat2fine.json --index splits/index_gdcld.csv --raw data --runs $RUNS --name X_gd_sat2fine_otsuM_s0 --init runs/C_sat2uav_M1_g0.5_s0/best.pt --seed 0 --gsd_target 0.5 --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0
# Bijie (pre-registered plan in splits/bijie_preregistration_20261022.md): SAT source retrained at the rule anchor 0.6 m, UAV source reused at 1.0 m.
# The Bijie adaptation runs excluded padded target pixels from the pseudo-label supervision (--tgt_pad_ignore 1); see README, section Notes.
python -m cas.train     --split splits/X_bj_sat.json --index splits/index_bijie.csv --raw data --runs $RUNS --name X_bj_sat_M1_g0.6_s0 $COMMON_SRC --seed 0 --gsd_target 0.6 --gsd_clamp 0.25,2.0
python -m cas.train_uda --split splits/X_bj_sat.json --index splits/index_bijie.csv --raw data --runs $RUNS --name X_bj_sat_otsuM_s0 --init runs/X_bj_sat_M1_g0.6_s0/best.pt --seed 0 --gsd_target 0.6 --gsd_clamp 0.25,2.0 $COMMON_UDA --pl_mode otsu --pl_gate margin --pl_margin 1.0 --tgt_pad_ignore 1

# --- Operating points, AP and object-level proxies (one pass per checkpoint) --------------------------------------
python -m cas.e3_infer --ckpt runs/C_sat2uav_M1_g0.5_s0/best.pt --split splits/C_sat2uav.json --index $IDX --raw $RAW --gsd_target 0.5 --out runs/E3/sat2uav_M1_s0
python scripts/e3_table.py
