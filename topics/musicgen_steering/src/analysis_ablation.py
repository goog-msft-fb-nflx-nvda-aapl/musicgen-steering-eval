"""pooled hit tables and paired comparisons for the 2026-10-19 ablation round (src/run_ablations.sh). reads results/ablation/<dir>/*_arms.json (03_arm_compare_v2 output).
pairs are (note, gen seed, prompt) restricted to the gen seeds present in both files being compared; exact mcnemar (no multiplicity correction here, ~8 tests per table: read p loosely).
usage: python analysis_ablation.py
"""
import json, glob, os, numpy as np
from scipy.stats import binomtest
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def load(path_fmt):
    P = {}
    for n in NOTES:
        for r in json.load(open(path_fmt.format(n=n)))["rows"]: P[(n, r["arm"], r["eta"], r["seed"], r["prompt"])] = r["hit"]
    return P
def table(P, title):
    etas = sorted({k[2] for k in P if k[1] != "base"}); print(f"\n{title}  (base {np.mean([v for k, v in P.items() if k[1] == 'base']):.3f}, n per cell {sum(1 for k in P if k[1] == 'const8' and k[2] == etas[0])})")
    for arm in ("const8", "musicrfm"): print(f"  {arm:9s}" + "  ".join(f"eta{e}: {np.mean([v for k, v in P.items() if k[1] == arm and k[2] == e]):.3f}" for e in etas))
def paired(A, B, name):
    ks = [k for k in A if k in B and k[1] != "base"]; out = []
    for arm in ("const8", "musicrfm"):
        for e in sorted({k[2] for k in ks}):
            kk = [k for k in ks if k[1] == arm and k[2] == e]; a = np.array([A[k] for k in kk]); b = np.array([B[k] for k in kk]); x, y = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0; out.append(f"{arm} {e}: {a.mean():.3f} vs {b.mean():.3f} (only-first {x}, only-second {y}) p={p:.1e}")
    print(f"  paired {name}:\n    " + "\n    ".join(out))
R = "results/ablation"; S = "results/arms"; M = "results/arms_medium"
# 1. medium, higher eta (meandiff, rfm) together with the earlier .15-.6 grid
for m in ("meandiff", "rfm"):
    hi = load(f"{R}/arms_medium_hi/musicgen_medium_{{n}}{'_meandiff' if m == 'meandiff' else ''}_arms.json"); lo = load(f"{M}/musicgen_medium_{{n}}{'_meandiff' if m == 'meandiff' else ''}_arms.json")
    table(lo, f"medium {m} eta .15-.6"); table(hi, f"medium {m} eta .9-2.0")
# 2. rfm eigendirection index (small); reference = top direction (comp 0), same gen seeds 0,1
ref = {k: v for k, v in load(f"{S}/musicgen_small_{{n}}_arms.json").items() if k[3] in (0, 1)}; table(ref, "small rfm comp 0 (top), seeds 0,1")
for c in (1, 2):
    P = load(f"{R}/arms_comp{c}/musicgen_small_{{n}}_arms.json"); table(P, f"small rfm comp {c}"); paired(ref, P, f"comp0 vs comp{c}")
# 3. training seeds (small): seed 42 reference vs 43 (44 when present)
for m in ("rfm", "meandiff"):
    ref = {k: v for k, v in load(f"{S}/musicgen_small_{{n}}{'' if m == 'rfm' else '_meandiff'}_arms.json").items() if k[3] in (0, 1)}; table(ref, f"small {m} train seed 42, gen seeds 0,1")
    for s in (43, 44):
        d = f"{R}/arms_seed{s}"
        if not glob.glob(f"{d}/musicgen_small_*{'_meandiff' if m == 'meandiff' else ''}_arms.json") or (m == "rfm" and len(glob.glob(f"{d}/*_arms.json")) < 12): continue
        try: P = load(f"{d}/musicgen_small_{{n}}{'' if m == 'rfm' else '_meandiff'}_arms.json")
        except FileNotFoundError: continue
        table(P, f"small {m} train seed {s}"); paired(ref, P, f"seed 42 vs {s}")
