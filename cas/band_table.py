"""Table 2 row: per-GSD-band IoU pooled over subsets, reconstructed from per-subset P/R and
landslide pixel counts of the test tiles (native resolution); multi-seed = mean over seeds.
usage: python -m cas.band_table <method> <seeds> [runs_dir]   e.g. advent 0,1,2,3,4
run names: C_{dir}_{method}_M1_s{seed} (DACS-style: C_{dir}_M1ST_g{G}_s{seed})."""
import csv, json, sys, statistics as st
from collections import defaultdict

BANDS = {
    'uav2sat': [('<=1 m', ['Longxi_SAT', 'Mengdong', 'Moxitaidi_SAT']),
                ('3-4 m', ['Hokkaido', 'Tiburon_Planet']),
                ('5 m', ['Lombok', 'Palu', 'Tiburon_Sentinel', 'Wenchuan'])],
    'sat2uav': [('0.2 m', ['Jiuzhai_UAV_0.2', 'Moxitown_UAV_0.2']),
                ('0.5-0.6 m', ['Jiuzhai_UAV_0.5', 'Longxi_UAV', 'Moxitaidi_UAV_0.6']),
                ('1 m', ['Moxitaidi_UAV_1', 'Moxitown_UAV_1'])],
}

def npos_by_sid():
    n = defaultdict(float)
    for r in csv.DictReader(open('splits/index.csv', newline='')):
        n[r['sid']] += float(r['ls_frac']) * int(r['h']) * int(r['w'])
    return n

def name(d, m, s):
    if m in ('dacs', 'M1ST'):
        return f"C_{d}_M1ST_g{'1.0' if d == 'uav2sat' else '0.5'}_s{s}"
    return f'C_{d}_{m}_M1_s{s}'

def row(d, m, seeds, runs, npos):
    test = set(json.load(open(f'splits/C_{d}.json'))['test'])
    per = []
    for s in seeds:
        a = json.load(open(f'{runs}/{name(d, m, s)}/metrics_test.json'))['at_0.5']
        cnt = {}
        for k, v in a.items():
            if not k.startswith('sid:'):
                continue
            sid = k[4:]
            P, R = v['Precision'], v['Recall']
            tp = R * npos[sid]; fn = npos[sid] - tp; fp = tp * (1 - P) / P if P > 0 else 0.0
            cnt[sid] = (tp, fp, fn)
        def iou(sids):
            tp = sum(cnt[x][0] for x in sids); fp = sum(cnt[x][1] for x in sids); fn = sum(cnt[x][2] for x in sids)
            return tp / (tp + fp + fn)
        missing = [x for _, ss in BANDS[d] for x in ss if x not in cnt]
        assert not missing, (missing, sorted(cnt))
        allsid = [x for _, ss in BANDS[d] for x in ss]
        rec = iou(allsid)
        assert abs(rec - a['ALL']['IoU']) < 5e-4, (s, rec, a['ALL']['IoU'])
        per.append([iou(ss) for _, ss in BANDS[d]] + [a['ALL']['Precision'], a['ALL']['Recall'], a['ALL']['F1'], a['ALL']['IoU']])
    mean = [st.mean(c) for c in zip(*per)]
    sd = st.stdev([p[-1] for p in per]) if len(per) > 1 else 0.0
    cells = ' | '.join(f'.{round(x * 1000):03d}' for x in mean[:-1])
    iou_s = f'.{round(mean[-1] * 1000):03d}' + (f' ± .{round(sd * 1000):03d}' if len(per) > 1 else '')
    print(f"{d} {m} seeds={len(per)} | {cells} | {iou_s} | per-seed IoU " + ' '.join(f'{p[-1]:.4f}' for p in per))

if __name__ == '__main__':
    m = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(',')]; runs = sys.argv[3] if len(sys.argv) > 3 else 'runs'
    npos = npos_by_sid()
    for d in ('uav2sat', 'sat2uav'):
        row(d, m, seeds, runs, npos)
