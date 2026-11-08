"""Build the evaluation protocols from splits/index.csv.

    python -m cas.splits [--index splits/index.csv] [--out splits] [--seed 0]

Writes JSON files, each {"train": [...], "val": [...], "test": [...]} with entries "sid/stem":
  A_random73.json            protocol A0: per-subset random 7:3, val==test   (paper / SC-Net convention, for comparability only)
  A_712.json                 protocol A : per-subset random 7:1:2 (train/val/test), mixed UAV+SAT  -> main in-domain table
  A_uav_712.json, A_sat_712.json   protocol A split by platform (separate models, as SC-Net / CAS paper report UAV and SAT)
  B_loro_<Region>.json       protocol B : leave-one-region-out, 9 folds (region groups defined in cas.meta)
  C_uav2sat.json / C_sat2uav.json   protocol C : cross-platform
For B and C a 10 % subset-stratified slice of the training pool is held out as 'val' for model selection,
so the held-out region / platform is never used for early stopping or threshold choice.
The 7:1:2 test tiles of A_712 are exactly a subset of the 7:3 test tiles of A_random73 (same shuffle, seed 0).
"""
import argparse, csv, json, os, random
from collections import defaultdict


def load(index):
    with open(index) as f:
        return list(csv.DictReader(f))


def key(r):
    return f"{r['sid']}/{r['stem']}"


def hold_out_val(train_rows, frac, rng):
    by_sid = defaultdict(list)
    for r in train_rows:
        by_sid[r["sid"]].append(r)
    tr, va = [], []
    for sid, rows in sorted(by_sid.items()):
        rows = rows[:]
        rng.shuffle(rows)
        k = max(1, int(round(len(rows) * frac)))
        va += rows[:k]; tr += rows[k:]
    return tr, va


def dump(path, tr, va, te, note):
    d = {"note": note, "n": {"train": len(tr), "val": len(va), "test": len(te)},
         "train": sorted(map(key, tr)), "val": sorted(map(key, va)), "test": sorted(map(key, te))}
    with open(path, "w") as f:
        json.dump(d, f)
    print(f"{os.path.basename(path):28s} train {len(tr):6d}  val {len(va):5d}  test {len(te):6d}   {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="splits/index.csv")
    ap.add_argument("--out", default="./splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    a = ap.parse_args()
    rows = load(a.index)

    # ---- protocol A family: one shuffle per subset (seed 0), then 7:3 and 7:1:2 cuts of the SAME order
    by_sid = defaultdict(list)
    for r in rows:
        by_sid[r["sid"]].append(r)
    shuffled = {}
    rng = random.Random(a.seed)
    for sid, rs in sorted(by_sid.items()):
        rs = rs[:]; rng.shuffle(rs); shuffled[sid] = rs
    tr73, te73, tr712, va712, te712 = [], [], [], [], []
    for sid, rs in sorted(shuffled.items()):
        n = len(rs); k70 = int(round(n * 0.7)); k80 = int(round(n * 0.8))
        tr73 += rs[:k70]; te73 += rs[k70:]
        tr712 += rs[:k70]; va712 += rs[k70:k80]; te712 += rs[k80:]
    dump(os.path.join(a.out, "A_random73.json"), tr73, te73, te73, "per-subset random 7:3, val==test (paper convention)")
    dump(os.path.join(a.out, "A_712.json"), tr712, va712, te712, "per-subset random 7:1:2, mixed UAV+SAT (main)")
    for plat in ("UAV", "SAT"):
        f = lambda rs: [r for r in rs if r["platform"] == plat]
        dump(os.path.join(a.out, f"A_{plat.lower()}_712.json"), f(tr712), f(va712), f(te712), f"per-subset random 7:1:2, {plat} only (separate model)")

    # ---- protocol B: leave-one-region-out
    regions = sorted({r["region"] for r in rows})
    for reg in regions:
        te = [r for r in rows if r["region"] == reg]
        pool = [r for r in rows if r["region"] != reg]
        tr, va = hold_out_val(pool, a.val_frac, random.Random(a.seed))
        dump(os.path.join(a.out, f"B_loro_{reg}.json"), tr, va, te, f"test region={reg}")

    # ---- protocol C: cross-platform
    for src, dst in (("UAV", "SAT"), ("SAT", "UAV")):
        pool = [r for r in rows if r["platform"] == src]
        te = [r for r in rows if r["platform"] == dst]
        tr, va = hold_out_val(pool, a.val_frac, random.Random(a.seed))
        dump(os.path.join(a.out, f"C_{src.lower()}2{dst.lower()}.json"), tr, va, te, f"train {src} -> test {dst}")


if __name__ == "__main__":
    main()
