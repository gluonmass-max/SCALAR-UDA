"""Bijie pre-registered outcomes over seeds 0-2 (read-only). IoU@0.5 on all 2773 crops from metrics_test.json
(adapted runs) and from runs/E3/<name>/summary.json (native and anchored source models); AP from E3 summaries.
Paired t-tests (n = 3, two-sided) on the per-seed differences. Usage: python scripts/bijie_stats.py"""
import json, os, math
import numpy as np
from scipy import stats
def iou_run(name):
    f=f'runs/{name}/metrics_test.json'
    return json.load(open(f))['at_0.5']['ALL']['IoU'] if os.path.exists(f) else None
def e3(name):
    f=f'runs/E3/{name}/summary.json'
    if not os.path.exists(f): return None, None
    d=json.load(open(f)); a=d['AP']
    if isinstance(a,dict):
        v=a.get('ALL',a)
        if isinstance(v,dict):
            v=next((v[k] for k in ('AP','ap','env','mean') if k in v), next(iter(v.values())))
        a=v
    return d['pixel']['t05']['ALL']['IoU'], float(a)
def ms(x): x=np.array([v for v in x if v is not None],float); return f'{x.mean():.4f} ± {x.std(ddof=1):.4f} (n={len(x)})' if len(x)>1 else (f'{x[0]:.4f}' if len(x) else '—')
def paired(a,b,label):
    d=np.array([(x-y) for x,y in zip(a,b) if x is not None and y is not None])*100
    if len(d)<2: print(f'  {label}: ' + ', '.join(f'{v:+.1f}' for v in d)); return
    t,p=stats.ttest_1samp(d,0.0)
    print(f'  {label}: per seed ' + ', '.join(f'{v:+.1f}' for v in d) + f' | mean {d.mean():+.1f} ± {d.std(ddof=1):.1f} | {int((d>0).sum())}/{len(d)} positive | paired t p = {p:.3f}')
for T,src,tag in [('T1 CAS SAT -> Bijie (anchor 0.6 m)','sat','bj_sat'),('T2 CAS UAV -> Bijie (anchor 1 m)','uav','bj_uav')]:
    print('==',T)
    nat,nat_ap=e3(f'{tag}_native_s0'); print(f'  native source (seed 0): IoU {nat:.4f} AP {nat_ap:.4f}')
    M1=[e3(f'{tag}_M1_s{s}')[0] for s in range(3)]; M1ap=[e3(f'{tag}_M1_s{s}')[1] for s in range(3)]
    LOC=[iou_run(f'X_bj_{src}_otsuM_s{s}') for s in range(3)]; LOCap=[e3(f'{tag}_otsuM_s{s}')[1] for s in range(3)]
    SC=[iou_run(f'X_bj_{src}_asym_s{s}') for s in range(3)]; SCap=[e3(f'{tag}_SCALAR_s{s}')[1] for s in range(3)]
    print('  Source+GSA IoU', ms(M1), '| AP', ms(M1ap)); print('  GSA+LOC    IoU', ms(LOC), '| AP', ms(LOCap)); print('  SCALAR     IoU', ms(SC), '| AP', ms(SCap))
    paired(SC,M1,'PRIMARY SCALAR - Source+GSA'); paired(LOC,M1,'LOC step'); paired(SC,LOC,'RSR step')
    paired(M1,[nat]*3,'GSA step vs native(seed0)'); paired(SC,[nat]*3,'SCALAR - native(seed0)')
    ind=iou_run(f'X_bj_{src}_asym_ind_s0'); print(f'  inductive half SCALAR (seed 0): {ind}')
