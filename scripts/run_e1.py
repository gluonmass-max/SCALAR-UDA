"""E1 runner: re-use the exact argument set of an original run (first line of its log.txt) with a few overrides.
  python scripts/run_e1.py <template log> <name> key=value [key=value ...]
Overrides always applied: index, raw, runs (machine-specific paths) and name. Values are strings except that 'None' removes the key.
Refuses to start if any template key is unknown to the current argparse of the chosen module, and skips runs that
already have metrics_test.json."""
import json, os, subprocess, sys

log, name = sys.argv[1], sys.argv[2]
a = json.loads(open(log).readline()[len("args "):])
mod = "cas.train_uda" if "uda_mode" in a else "cas.train"
a.update(index="splits/index.csv", raw="data/CAS_raw", runs="runs", name=name)
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    if v == "None":
        a.pop(k, None)
    else:
        a[k] = v
if os.path.exists(os.path.join("runs", name, "metrics_test.json")):
    print(f"skip {name}: metrics_test.json exists", flush=True); sys.exit(0)
helptxt = subprocess.run([sys.executable, "-m", mod, "--help"], capture_output=True, text=True).stdout
unknown = [k for k in a if f"--{k}" not in helptxt]
if unknown:
    print(f"ABORT {name}: unknown args for {mod}: {unknown}", flush=True); sys.exit(2)
cmd = [sys.executable, "-m", mod] + [x for k, v in a.items() for x in (f"--{k}", str(v))]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
