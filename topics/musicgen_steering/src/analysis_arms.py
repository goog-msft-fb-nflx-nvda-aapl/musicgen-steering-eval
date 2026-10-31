"""paired analysis of results/arms/*_arms.json (output of 03_arm_compare.py), all 12 notes, rfm directions.
per note x arm x eta: hit rate, discordant pairs vs the unsteered baseline; pooled over notes: exact McNemar (Holm over 8 cells),
arm-vs-arm pooled McNemar. pairs are (note, gen seed, prompt) and are not independent draws of prompts (20 prompts reused).
usage: python analysis_arms.py   (run from repo root; cpu, reads json only)
"""
import json
import numpy as np
from scipy.stats import binomtest
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]; ETAS = [0.15, 0.3, 0.45, 0.6]; ARMS = ["const8", "musicrfm"]
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); m = len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
P = {}
for n in NOTES:
    for r in json.load(open(f"results/arms/musicgen_small_{n}_arms.json"))["rows"]: P[(n, r["arm"], r["eta"], r["seed"], r["prompt"])] = r
keys = [(n, s, p) for n in NOTES for s in range(4) for p in range(20)]
def hits(n_list, arm, e): return np.array([P[(n, arm, e, s, p)]["hit"] for n, s, p in keys if n in n_list])
print("base hit rate per note:", {n: round(float(hits([n], "base", 0.0).mean()), 3) for n in NOTES})
for arm in ARMS:
    print(f"\n{arm}: hit rate by note, eta {ETAS}  (discordant up/down vs base at eta=0.6)")
    for n in NOTES:
        b = hits([n], "base", 0.0)
        c6 = hits([n], arm, 0.6)
        print(f"{n:3s}", " ".join(f"{hits([n], arm, e).mean():.2f}" for e in ETAS), f"| 0.6: up {int(((b == 0) & (c6 == 1)).sum())} down {int(((b == 1) & (c6 == 0)).sum())}")
print("\npooled over 12 notes (960 pairs per cell), exact McNemar vs base, Holm over 8 cells")
rows = []
for arm in ARMS:
    for e in ETAS:
        b, c = hits(NOTES, "base", 0.0), hits(NOTES, arm, e)
        up, dn = int(((b == 0) & (c == 1)).sum()), int(((b == 1) & (c == 0)).sum())
        rows.append((arm, e, c.mean(), b.mean(), up, dn, binomtest(up, up + dn, .5).pvalue if up + dn else 1.0))
for r, a in zip(rows, holm([r[6] for r in rows])): print(f"{r[0]:9s} eta {r[1]:<4} hit {r[2]:.3f} vs base {r[3]:.3f} up {r[4]} down {r[5]} p={r[6]:.2e} holm={a:.2e}")
print("\narm vs arm pooled (musicrfm vs const8):")
rows2 = []
for e in ETAS:
    a, b = hits(NOTES, "const8", e), hits(NOTES, "musicrfm", e)
    u, d = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()); rows2.append((e, a.mean(), b.mean(), u, d, binomtest(u, u + d, .5).pvalue if u + d else 1.0))
for r, h in zip(rows2, holm([r[5] for r in rows2])): print(f"eta {r[0]:<4} const8 {r[1]:.3f} musicrfm {r[2]:.3f} (rfm-only {r[3]}, const-only {r[4]}) p={r[5]:.2e} holm={h:.2e}")
