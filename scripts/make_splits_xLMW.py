"""E1 splits (2026-10-19): region-held-out full SCALAR.
X = {Longxi, Moxi, Wenchuan} is removed from the source and from the adaptation pool; Longxi and Moxi target tiles form a
separate held-out evaluation set. Wenchuan is removed because the CAS study-area polygon of Longxi River lies inside the
Wenchuan polygon. Validation hold-out uses the same per-subset 10 % rule and seed as splits.py.
Also writes the merged LORO fold B_loro_LongxiWenchuan.json (test = Longxi + Wenchuan, 10 % validation from the rest).
  python make_splits_xLMW.py --index splits/index.csv --out splits
"""
import argparse, csv, json, os, random
from collections import Counter, defaultdict

X = {"Longxi", "Moxi", "Wenchuan"}
HOLD = {"Longxi", "Moxi"}


def key(r):
    return f"{r['sid']}/{r['stem']}"


def hold_out_val(train_rows, frac, rng):          # identical to splits.py
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
    sub = Counter(r["sid"] for r in te)
    print(f"{os.path.basename(path):34s} train {len(tr):5d}  val {len(va):4d}  test {len(te):5d}  "
          f"regions(test)={sorted({r['region'] for r in te})}  {note}")
    print("   test subsets:", dict(sorted(sub.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="splits/index.csv"); ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val_frac", type=float, default=0.1)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.index)))
    assert X <= {r["region"] for r in rows}, "region names not found in index"
    for src, dst in (("UAV", "SAT"), ("SAT", "UAV")):
        d = f"{src.lower()}2{dst.lower()}"
        pool_src = [r for r in rows if r["platform"] == src and r["region"] not in X]
        tr, va = hold_out_val(pool_src, a.val_frac, random.Random(a.seed))
        te = [r for r in rows if r["platform"] == dst and r["region"] not in X]
        ho = [r for r in rows if r["platform"] == dst and r["region"] in HOLD]
        assert not ({r["region"] for r in tr + va} & {r["region"] for r in ho}), "source/held-out region overlap"
        assert not ({r["region"] for r in te} & {r["region"] for r in ho}), "pool/held-out region overlap"
        dump(os.path.join(a.out, f"C_xLMW_{d}.json"), tr, va, te,
             f"E1: {src} source without Longxi/Moxi/Wenchuan -> {dst} pool without them (transductive)")
        dump(os.path.join(a.out, f"C_xLMW_{d}_holdout.json"), tr, va, ho,
             f"E1: same source; test = Longxi + Moxi {dst} tiles (never in source or pool; eval only)")
    te = [r for r in rows if r["region"] in {"Longxi", "Wenchuan"}]
    pool = [r for r in rows if r["region"] not in {"Longxi", "Wenchuan"}]
    tr, va = hold_out_val(pool, a.val_frac, random.Random(a.seed))
    dump(os.path.join(a.out, "B_loro_LongxiWenchuan.json"), tr, va, te,
         "LORO merged fold: test region = Longxi + Wenchuan (Longxi polygon lies inside the Wenchuan polygon)")


if __name__ == "__main__":
    main()
