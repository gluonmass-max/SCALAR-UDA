"""CAS Landslide dataset metadata: subset -> region / platform / sensor / GSD / licence.
Zip names are exactly as uploaded to data/CAS; sid is the canonical folder name under data/CAS_raw.
Counts and GSDs follow Table 2 of Xu et al., Sci. Data 11, 12 (2024) and the Zenodo README.
Inside every zip: img/*.tif (3x512x512 uint8 RGB), mask/*.tif (1x512x512 uint8, 0/1), label/*.tif (RGB 0/255 visualisation, unused).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Subset:
    sid: str
    zip_name: str
    region: str
    platform: str  # 'UAV' | 'SAT'
    sensor: str
    gsd_m: float
    n_expected: int
    licence: str


SUBSETS = [
    Subset("Palu",              "palu.zip",                        "Palu",     "SAT", "WorldView2/3", 5.0, 817,  "CC BY-NC 4.0"),
    Subset("Lombok",            "Lombok.zip",                      "Lombok",   "SAT", "WorldView2/3", 5.0, 436,  "CC BY-NC 4.0"),
    Subset("Hokkaido",          "Hokkaido Iburi-Tobu.zip",         "Hokkaido", "SAT", "GSI aerial",   3.0, 1484, "CC BY 4.0"),
    Subset("Tiburon_Sentinel",  "Tiburon Peninsula（Sentinel）.zip", "Tiburon",  "SAT", "Sentinel-2",   5.0, 606,  "Non-commercial"),
    Subset("Tiburon_Planet",    "Tiburon Peninsula（planet）.zip",   "Tiburon",  "SAT", "Planet",       4.0, 325,  "Planet E&R"),
    Subset("Mengdong",          "Mengdong Township.zip",           "Mengdong", "SAT", "SuperView-1",  0.5, 1155, "Image licence"),
    Subset("Moxitaidi_SAT",     "Moxitaidi (SAT).zip",             "Moxi",     "SAT", "Sentinel-2",   0.6, 652,  "Non-commercial"),
    Subset("Moxitaidi_UAV_0.6", "Moxitaidi (UAV-0.6m).zip",        "Moxi",     "UAV", "UAV",          0.6, 984,  "Derivative works"),
    Subset("Moxitaidi_UAV_1",   "Moxitaidi (UAV-1m).zip",          "Moxi",     "UAV", "UAV",          1.0, 483,  "Derivative works"),
    Subset("Moxitown_UAV_0.2",  "Moxi town（UAV-0.2m）.zip",         "Moxi",     "UAV", "UAV",          0.2, 1635, "Derivative works"),
    Subset("Moxitown_UAV_1",    "Moxi town（UAV-1m）.zip",           "Moxi",     "UAV", "UAV",          1.0, 160,  "Derivative works"),
    Subset("Longxi_SAT",        "Longxi River(SAT).zip",           "Longxi",   "SAT", "GF-1",         0.5, 1769, "Image licence"),
    Subset("Longxi_UAV",        "Longxi River（UAV）.zip",           "Longxi",   "UAV", "UAV",          0.5, 2504, "Derivative works"),
    Subset("Jiuzhai_UAV_0.2",   "Jiuzhai valley (UAV-0.2m).zip",   "Jiuzhai",  "UAV", "UAV",          0.2, 5925, "Derivative works"),
    Subset("Jiuzhai_UAV_0.5",   "Jiuzhai valley (UAV-0.5m).zip",   "Jiuzhai",  "UAV", "UAV",          0.5, 1752, "Derivative works"),
    Subset("Wenchuan",          "汶川地震-178.zip",                  "Wenchuan", "SAT", "Landsat",      5.0, 178,  "Image licence"),
]

BY_ID = {s.sid: s for s in SUBSETS}
REGIONS = sorted({s.region for s in SUBSETS})  # Hokkaido Jiuzhai Lombok Longxi Mengdong Moxi Palu Tiburon Wenchuan
