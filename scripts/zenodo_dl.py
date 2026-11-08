#!/usr/bin/env python3
"""Multi-range resumable downloader for Zenodo files (no aria2c on this box).
Usage: zenodo_dl.py <record_id> <out_dir> [--only name1,name2] [--conns 16]
Downloads every file in the record except those excluded; resumable via .part files."""
import json, os, sys, time, urllib.request, urllib.parse, threading, argparse
ap = argparse.ArgumentParser(); ap.add_argument('record'); ap.add_argument('out'); ap.add_argument('--only', default=''); ap.add_argument('--skip', default=''); ap.add_argument('--conns', type=int, default=16); ap.add_argument('--chunk', type=int, default=8*1024*1024)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rec = json.load(urllib.request.urlopen(f'https://zenodo.org/api/records/{a.record}', timeout=60))
files = rec['files']; only = set(x for x in a.only.split(',') if x); skip = set(x for x in a.skip.split(',') if x)
def safe(n): return n.replace(' ', '_').replace('（', '(').replace('）', ')')
for f in files:
    name = f['key']
    if (only and name not in only) or name in skip: continue
    url = f['links']['self']; size = f['size']; out = os.path.join(a.out, safe(name)); part = out + '.part'; done_f = out + '.done'
    if os.path.exists(out) and os.path.getsize(out) == size: print('skip', name, flush=True); continue
    print(f'== {name} {size/1e6:.1f} MB', flush=True)
    with open(part, 'ab') as fh: fh.truncate(size)
    chunks = list(range(0, size, a.chunk)); done = set()
    if os.path.exists(done_f): done = set(int(x) for x in open(done_f).read().split())
    todo = [c for c in chunks if c not in done]; lock = threading.Lock(); t0 = time.time(); got = [0]
    def worker():
        while True:
            with lock:
                if not todo: return
                s = todo.pop(0)
            e = min(s + a.chunk, size) - 1
            for attempt in range(20):
                try:
                    req = urllib.request.Request(url, headers={'Range': f'bytes={s}-{e}'})
                    data = urllib.request.urlopen(req, timeout=90).read()
                    if len(data) != e - s + 1: raise IOError('short read')
                    with lock:
                        with open(part, 'r+b') as fh: fh.seek(s); fh.write(data)
                        with open(done_f, 'a') as df: df.write(f'{s}\n')
                        got[0] += len(data)
                    break
                except Exception as ex:
                    time.sleep(min(60, 3 * (attempt + 1)))
            else:
                with lock: todo.append(s)
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(a.conns)]
    [t.start() for t in ths]
    while any(t.is_alive() for t in ths):
        time.sleep(30); el = time.time() - t0
        print(f'   {name}: {(len(done)*a.chunk+got[0])/1e6:.0f}/{size/1e6:.0f} MB  {got[0]/el/1e6:.2f} MB/s', flush=True)
    os.replace(part, out); os.remove(done_f); print('done', name, f'{time.time()-t0:.0f}s', flush=True)
print('ALL DONE', flush=True)
