import numpy as np
from scipy import stats
gsa_p=[.3058,.2485,.3154,.2508,.2513]; gsa_h=[.2714,.1349,.2476,.1647,.1286]
sc_p=[.3744,.2828,.3890,.3297,.2847]; sc_h=[.4291,.2127,.4030,.2049,.1738]
ms=lambda x: f"{np.mean(x):.3f} ± {np.std(x,ddof=1):.3f}"
for lab,a,b in (("pool",gsa_p,sc_p),("hold",gsa_h,sc_h)):
    d=np.array(b)-np.array(a); t,p=stats.ttest_rel(b,a)
    print(f"{lab}: Source+GSA {ms(a)}  SCALAR {ms(b)}  diff {d.mean()*100:+.1f} ± {d.std(ddof=1)*100:.1f} pts, pos {(d>0).sum()}/5, p={p:.4f}")
