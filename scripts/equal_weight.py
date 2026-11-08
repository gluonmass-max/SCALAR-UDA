"""R2-M8: subset-level equal-weight averages from E3 summary.json (pixel metrics per sid at threshold 0.5), next to pooled IoU."""
import json, os, sys, statistics as st
R = sys.argv[1]
groups = {
  "sat2uav": {"原生源 s0": ["sat2uav_native_s0"], "Source+GSA s0–4": [f"sat2uav_M1_s{s}" for s in range(5)], "GSA+LOC s0": ["sat2uav_otsuM_s0"], "SCALAR s0": ["sat2uav_SCALAR_s0"],
              "DACS s1–4": [f"sat2uav_DACS_s{s}" for s in range(1,5)], "FDA s1–4": [f"sat2uav_FDA_s{s}" for s in range(1,5)], "LandsDA s1–4": [f"sat2uav_LandsDA_s{s}" for s in range(1,5)], "MIC s1–4": [f"sat2uav_MIC_s{s}" for s in range(1,5)]},
  "uav2sat": {"原生源 s0": ["uav2sat_native_s0"], "Source+GSA s0–4": [f"uav2sat_M1_s{s}" for s in range(5)], "GSA+LOC s0": ["uav2sat_otsuM_s0"], "SCALAR s0": ["uav2sat_SCALAR_s0"],
              "DACS s1–3": [f"uav2sat_DACS_s{s}" for s in range(1,4)], "FDA s1–4": [f"uav2sat_FDA_s{s}" for s in range(1,5)], "LandsDA s1–4": [f"uav2sat_LandsDA_s{s}" for s in range(1,5)], "MIC s1–3": [f"uav2sat_MIC_s{s}" for s in range(1,4)]},
}
def load(n):
    p = os.path.join(R, n, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None
print("| 方向 | 方法 | 像素汇总 IoU | 子集等权 IoU | 区域等权 IoU | 子集数 | 最低子集 |")
print("|---|---|---|---|---|---|---|")
for d, fams in groups.items():
    for name, runs in fams.items():
        pooled, sub, reg, nsub, worst = [], [], [], 0, []
        for n in runs:
            s = load(n)
            if s is None: continue
            px = s["pixel"]["t05"]
            pooled.append(px["ALL"]["IoU"])
            sids = {k: v["IoU"] for k, v in px.items() if k.startswith("sid:")}
            regs = {k: v["IoU"] for k, v in px.items() if k.startswith("region:")}
            sub.append(st.mean(sids.values())); reg.append(st.mean(regs.values())); nsub = len(sids)
            worst.append(min(sids.items(), key=lambda kv: kv[1]))
        if not pooled: continue
        w = min(worst, key=lambda kv: kv[1])
        print(f"| {'SAT→UAV' if d=='sat2uav' else 'UAV→SAT'} | {name} | {st.mean(pooled):.3f} | {st.mean(sub):.3f} | {st.mean(reg):.3f} | {nsub} | {w[0][4:]} {w[1]:.3f} |")
# E1 hold-out: Longxi / Moxi equal weight
print()
print("| E1 留出 | 方法 | 像素汇总 | 区域等权（Longxi, Moxi） |"); print("|---|---|---|---|")
for d in ("sat2uav", "uav2sat"):
    for name, runs in {"Source+GSA s0–4": [f"E1_{d}_M1_g0.5_s{s}_hold" for s in range(5)], "SCALAR s0–4": [f"E1_{d}_asym_s{s}_hold" for s in range(5)], "GSA+LOC s0": [f"E1_{d}_otsuM_s0_hold"], "原生源 s0": [f"E1_{d}_src_s0_hold"]}.items():
        pooled, reg = [], []
        for n in runs:
            s = load(n)
            if s is None: continue
            px = s["pixel"]["t05"]; pooled.append(px["ALL"]["IoU"]); reg.append(st.mean(v["IoU"] for k, v in px.items() if k.startswith("region:")))
        if pooled: print(f"| {'SAT→UAV' if d=='sat2uav' else 'UAV→SAT'} | {name} | {st.mean(pooled):.3f} | {st.mean(reg):.3f} |")
