"""Table A7(a) ablation rows (10-23): pixel-pooled IoU, unweighted mean over subsets and over regions, per seed, for the
five main-protocol runs of each ablation model; mean ± sample sd across seeds. Reads metrics_test.json of each run
(per-sid / per-region IoU computed by cas.train from integer counts). Run from the repository root: python scripts/equal_weight_5seeds.py"""
import json, os, statistics as st
def load(run):
    for base in ['runs']:
        p = f'{base}/{run}/metrics_test.json'
        if os.path.exists(p): return json.load(open(p))['at_0.5']
    return None
fams = {'uav2sat': {'native': 'C_uav2sat_segformer_b2_s{s}', 'Source+GSA': 'C_uav2sat_M1_g1.0_s{s}',
                    'GSA+LOC': 'C_uav2sat_M1ST_otsuM_s{s}', 'SCALAR': 'C_uav2sat_M1ST_asym_p15n05_s{s}'},
        'sat2uav': {'native': 'C_sat2uav_segformer_b2_s{s}', 'Source+GSA': 'C_sat2uav_M1_g0.5_s{s}',
                    'GSA+LOC': 'C_sat2uav_M1ST_otsuM_s{s}', 'SCALAR': 'C_sat2uav_M1ST_asym_p05n15_s{s}'}}
f = lambda x: f'{st.mean(x):.4f} ± {st.stdev(x):.4f}'
for d, fam in fams.items():
    print('==', d)
    for lab, pat in fam.items():
        pooled, sub, reg = [], [], []
        for s in range(5):
            a = load(pat.format(s=s)); assert a is not None, pat.format(s=s)
            sids = [v['IoU'] for k, v in a.items() if k.startswith('sid:')]
            regs = [v['IoU'] for k, v in a.items() if k.startswith('region:')]
            pooled.append(a['ALL']['IoU']); sub.append(sum(sids) / len(sids)); reg.append(sum(regs) / len(regs))
            print(f'   {lab:11s} seed {s}: pooled {pooled[-1]:.4f} subset-equal {sub[-1]:.4f} region-equal {reg[-1]:.4f}')
        print(f'{lab:11s} pooled {f(pooled)} | subset-equal {f(sub)} | region-equal {f(reg)} | n_sid {len(sids)} n_reg {len(regs)}')
