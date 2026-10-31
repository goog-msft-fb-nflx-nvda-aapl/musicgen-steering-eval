"""Audiobox Aesthetics (PQ, PC, CE, CU) of the extra-decade generations: per-song change versus the neutral prompt of the same set (seed means), Wilcoxon over songs; neutral PQ for reference. usage: python 79b_extra_quality_table.py tag ... -> results/extra_quality.txt (inputs results/quality_<tag>.json from 18_quality.py with ERAS=neutral,1970s,1990s,2000s)"""
import sys, json, numpy as np
from scipy.stats import wilcoxon
txt = []
for t in sys.argv[1:]:
    Q = json.load(open(f"results/quality_{t}.json")); songs = sorted({n.split("|")[0] for n in Q if n.split("|")[1] == "neutral"}); line = [t + f" (n_songs={len(songs)}); neutral PQ {np.mean([np.mean([Q[f'{s}|neutral|{sd}']['PQ'] for sd in (0, 1, 2)]) for s in songs]):.2f}"]
    for k in ("PQ", "CE"):
        cells = []
        for e in ("1970s", "1990s", "2000s"):
            f = lambda c: np.array([np.mean([Q[f"{s}|{c}|{sd}"][k] for sd in (0, 1, 2)]) for s in songs]); d = f(e) - f("neutral"); cells.append(f"{e} {d.mean():+.2f} [{int((d > 0).sum())}/{len(songs)}] p={wilcoxon(d).pvalue:.4f}")
        line.append(f"  {k} vs neutral: " + " | ".join(cells))
    txt += line
open("results/extra_quality.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
