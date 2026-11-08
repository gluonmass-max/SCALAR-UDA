# CAS Landslide evaluation protocols (frozen 2026-09-03, seed 0, cas/splits.py)

Unit: 512x512 tile (no georeference in CAS -> no spatial blocking possible inside a subset). Regions (cas/meta.py):
Palu | Lombok | Hokkaido | Tiburon {Sentinel, Planet} | Mengdong | Moxi {Moxitaidi SAT, UAV-0.6, UAV-1, Moxitown UAV-0.2, UAV-1} | Longxi {SAT, UAV} | Jiuzhai {UAV-0.2, UAV-0.5} | Wenchuan.
Platforms: UAV 13443 tiles (7 subsets), SAT 7422 tiles (9 subsets). Total 20865 (public Zenodo release; README's 22049 includes a non-public "Online" subset).

| id | file | train | val | test | rule | purpose |
|---|---|---|---|---|---|---|
| A0 | A_random73.json | 14604 | =test | 6261 | per-subset random 7:3, val==test | comparability with CAS paper / SC-Net (inflated by spatial leakage + test-based model selection) |
| A  | A_712.json | 14604 | 2088 | 4173 | same shuffle, per-subset 7:1:2, mixed UAV+SAT | main in-domain table; test tiles are a subset of A0 test |
| A-UAV | A_uav_712.json | 9410 | 1344 | 2689 | A restricted to UAV subsets, separate model | compare with per-platform numbers of CAS paper / SC-Net |
| A-SAT | A_sat_712.json | 5194 | 744 | 1484 | A restricted to SAT subsets, separate model | idem |
| B  | B_loro_<Region>.json x9 | 11869-18619 | 10% of pool, subset-stratified | region tiles (178-7677) | leave-one-region-out | true cross-region generalisation; gap to A = leakage inflation |
| C1 | C_uav2sat.json | 12100 | 1343 | 7422 | train UAV (10% val), test all SAT | cross-sensor |
| C2 | C_sat2uav.json | 6679 | 743 | 13443 | train SAT (10% val), test all UAV | cross-sensor |

Rules: model selection (best epoch, threshold) only on val; test touched once. Metrics: landslide IoU/F1/P/R pixel-pooled over the split, plus per-subset / per-region / per-platform. Report 3 seeds for main rows.
External zero-shot (no CAS-external training): GDCLD test (7 scenes; Palu overlaps CAS Palu -> report with and without), GDCLD val (4459 tiles), GVLM v1.1 (7 sites, post-event image).
