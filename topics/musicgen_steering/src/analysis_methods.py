"""paired comparison of direction methods (rfm, logistic, lda, meandiff) x arms (const8, musicrfm-style) x eta, pooled over 12 notes.
reads results/arms/musicgen_small_<note>[_<method>]_arms.json. pairs = (note, gen seed, prompt); every method used the same prompts and
seeds, so unsteered baselines should be identical across methods (checked below). exact McNemar vs unsteered (Holm over all cells)
and vs the rfm direction at the same arm/eta (Holm over method x arm x eta cells). cpu only.
"""
import json, sys
import numpy as np
from scipy.stats import binomtest
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]; ETAS = [0.15, 0.3, 0.45, 0.6]; ARMS = ["const8", "musicrfm"]
METHODS = ["rfm", "logistic", "lda", "meandiff"]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "small"; DIR = "results/arms" if MODEL == "small" else f"results/arms_{MODEL}"  # usage: python analysis_methods.py [small|medium]
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); m = len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
P = {}
for m in METHODS:
    for n in NOTES:
        f = f"{DIR}/musicgen_{MODEL}_{n}_arms.json" if m == "rfm" else f"{DIR}/musicgen_{MODEL}_{n}_{m}_arms.json"
        for r in json.load(open(f))["rows"]: P[(m, n, r["arm"], r["eta"], r["seed"], r["prompt"])] = r
keys = [(n, s, p) for n in NOTES for s in sorted({k[4] for k in P}) for p in range(20)]  # seeds present in the files (small: 0-3, medium: 0-1)
H = lambda m, arm, e: np.array([P[(m, n, arm, e, s, p)]["hit"] for n, s, p in keys]); F = lambda m, arm, e: np.array([P[(m, n, arm, e, s, p)]["frac"] for n, s, p in keys])
same = all(np.array_equal(H("rfm", "base", 0.0), H(m, "base", 0.0)) for m in METHODS)
print("unsteered baseline identical across method runs (hit vectors):", same, "| base hit rate", round(float(H("rfm", "base", 0.0).mean()), 3))
cells = [(m, a, e) for m in METHODS for a in ARMS for e in ETAS]
b = H("rfm", "base", 0.0); rows = []
for m, a, e in cells:
    c = H(m, a, e); up, dn = int(((b == 0) & (c == 1)).sum()), int(((b == 1) & (c == 0)).sum())
    rows.append(binomtest(up, up + dn, .5).pvalue if up + dn else 1.0)
adj = holm(rows)
print(f"\nhit rate pooled over 12 notes ({len(keys)} pairs), and Holm p vs unsteered over all 32 cells")
print("method   arm       " + "  ".join(f"eta{e:<5}" for e in ETAS))
for m in METHODS:
    for a in ARMS:
        print(f"{m:9s}{a:10s}" + "  ".join(f"{H(m, a, e).mean():.3f}   " for e in ETAS), " maxholm=%.1e" % max(adj[cells.index((m, a, e))] for e in ETAS))
print("\nD frac (target share of chroma energy) pooled mean minus unsteered:", )
for m in METHODS:
    for a in ARMS: print(f"{m:9s}{a:10s}" + "  ".join(f"{(F(m, a, e) - F('rfm', 'base', 0.0)).mean():+.3f}  " for e in ETAS))
print("\nvs rfm direction, same arm/eta (exact McNemar on discordant pairs, Holm over 24 cells)")
cmp = [(m, a, e) for m in METHODS[1:] for a in ARMS for e in ETAS]; ps = []; info = []
for m, a, e in cmp:
    x, y = H("rfm", a, e), H(m, a, e); u, d = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    ps.append(binomtest(u, u + d, .5).pvalue if u + d else 1.0); info.append((m, a, e, x.mean(), y.mean(), u, d))
for i, h in zip(info, holm(ps)): print(f"{i[0]:9s}{i[1]:10s} eta {i[2]:<4} rfm {i[3]:.3f} {i[0]} {i[4]:.3f} (method-only {i[5]}, rfm-only {i[6]}) holm={h:.1e}")
print("\nper-note hit rate at eta 0.6, musicrfm-style arm:")
print("note  " + "  ".join(f"{m:8s}" for m in METHODS))
for n in NOTES: print(f"{n:5s} " + "  ".join(f"{H(m, 'musicrfm', 0.6)[[i for i, k in enumerate(keys) if k[0] == n]].mean():.2f}    " for m in METHODS))
print("per-note hit rate at eta 0.6, const8 arm:")
for n in NOTES: print(f"{n:5s} " + "  ".join(f"{H(m, 'const8', 0.6)[[i for i, k in enumerate(keys) if k[0] == n]].mean():.2f}    " for m in METHODS))
