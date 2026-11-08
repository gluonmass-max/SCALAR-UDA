import numpy as np
from scipy import stats
D = {
 "sat2uav": {
  "gsa_pool":  [.1575,.1693,.1552,.1091,.1685], "gsa_hold": [.1459,.1255,.1526,.1180,.1045],
  "sc_pool":   [.3931,.3063,.3788,.3613,.3274], "sc_hold":  [.4066,.2750,.3904,.3971,.2989],
  "gsa_L": [.2320,.1469,.2096,.1654,.1482], "gsa_M": [.0700,.1075,.1035,.0778,.0675],
  "sc_L":  [.4055,.2666,.4098,.4315,.3266], "sc_M":  [.4075,.2822,.3731,.3668,.2745],
 },
 "uav2sat": {
  "gsa_pool":  [.3180,.3246,.2973,.3162,.3118], "gsa_hold": [.3645,.4236,.2604,.3772,.3822],
  "sc_pool":   [.3059,.3377,.3432,.2900,.3105], "sc_hold":  [.3215,.3839,.3167,.3680,.3664],
  "gsa_L": [.3904,.4631,.2785,.3918,.4006], "gsa_M": [.2744,.2689,.2009,.3232,.3248],
  "sc_L":  [.3192,.4018,.2952,.3718,.3918], "sc_M":  [.3277,.3441,.3774,.3586,.3172],
 }}
def ms(x): x=np.array(x); return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
for d, v in D.items():
    print(f"== {d}")
    for k in ["gsa_pool","sc_pool","gsa_hold","sc_hold","gsa_L","sc_L","gsa_M","sc_M"]:
        print(f"  {k:9s} {ms(v[k])}")
    for a,b,lab in [("gsa_pool","sc_pool","pool"),("gsa_hold","sc_hold","hold"),("gsa_L","sc_L","Longxi"),("gsa_M","sc_M","Moxi")]:
        diff = np.array(v[b])-np.array(v[a])
        t,p = stats.ttest_rel(v[b], v[a]); w = stats.wilcoxon(v[b], v[a], alternative="two-sided")
        print(f"  diff {lab:6s} {diff.mean()*100:+.1f} ± {diff.std(ddof=1)*100:.1f} pts  pos {int((diff>0).sum())}/5  t p={p:.4f}  wilcoxon p={w.pvalue:.3f}  per-seed {[f'{x*100:+.1f}' for x in diff]}")
