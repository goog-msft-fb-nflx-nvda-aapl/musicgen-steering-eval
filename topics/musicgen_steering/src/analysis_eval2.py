"""second-evaluator analysis of arms_eval2/*_eval2.npz (10b_second_eval.py): per method x arm x eta pooled over 12 notes (480 pairs/cell),
hit by the chroma argmax evaluator vs the MERT-v1-95M note probe (10a_mert_probe.py), per-clip agreement, and whether the method ordering holds under both.
usage: python analysis_eval2.py <dir with npz> <out json>"""
import sys, json, itertools, numpy as np
from scipy.stats import spearmanr
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]; METH = ["rfm", "meandiff", "logistic", "lda"]
d, outj = sys.argv[1], sys.argv[2]; MODEL = sys.argv[3] if len(sys.argv) > 3 else "small"; import os; METH = [m for m in METH if os.path.exists(f"{d}/musicgen_{MODEL}_C_{m}_eval2.npz")]; cells = {}; base = []; A = []
for m in METH:
    for n in NOTES:
        R = json.loads(str(np.load(f"{d}/musicgen_{MODEL}_{n.replace('#', 'sharp')}_{m}_eval2.npz")["rows"]))
        for r in R:
            (base if r["arm"] == "base" else cells.setdefault((m, r["arm"], r["eta"]), [])).append((r["hit"], r["mert_hit"], r["mert_p"]))
            A.append((r["hit"], r["mert_hit"]))
A = np.array(A); b = np.array(base)
print("clips", len(A), "chroma/mert hit agreement", round(float((A[:, 0] == A[:, 1]).mean()), 3), "| unsteered hit chroma", round(float(b[:, 0].mean()), 3), "mert", round(float(b[:, 1].mean()), 3), "n", len(b))
res = {}; print(f"{'method':9s} {'arm':9s} eta   chroma  mert   mert_p(target)")
for k in sorted(cells):
    v = np.array(cells[k]); res["|".join(map(str, k))] = dict(chroma=float(v[:, 0].mean()), mert=float(v[:, 1].mean()), mert_p=float(v[:, 2].mean()), n=len(v))
    print(f"{k[0]:9s} {k[1]:9s} {k[2]:.2f} {v[:, 0].mean():.3f}  {v[:, 1].mean():.3f}  {v[:, 2].mean():.3f}")
ks = sorted(cells); rho = spearmanr([res["|".join(map(str, k))]["chroma"] for k in ks], [res["|".join(map(str, k))]["mert"] for k in ks])[0]
print("spearman over", len(ks), "cells chroma vs mert:", round(rho, 3))
for arm in ("const8", "musicrfm"):
    for e in (0.15, 0.3, 0.45, 0.6):
        o = lambda key: [m for m, _ in sorted(((m, res[f"{m}|{arm}|{e}"][key]) for m in METH), key=lambda t: -t[1])]
        print(arm, e, "order chroma", o("chroma"), "| mert", o("mert"))
json.dump(res, open(outj, "w"), indent=1)
