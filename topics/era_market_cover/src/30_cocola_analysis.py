"""Summary of results/cocola.json: COCOLA score between the original vocal stem and an accompaniment (higher = more coherent), per mode; floor = same accompaniment against another song's vocal.
fit margin = matched - mismatched. paired Wilcoxon over songs (seeds averaged). usage: python 30_cocola_analysis.py [cocola.json]"""
import json, sys, numpy as np
from scipy.stats import wilcoxon
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/cocola.json")); songs = sorted({r["clip"] for r in R}); conds = ["original", "original_codec", "neutral", "1960s", "1980s", "2010s"]
def ps(c, k): return np.array([np.mean([r[k] for r in R if r["clip"] == s and r["cond"] == c]) for s in songs])
print(f"{len(songs)} songs, COCOLA_HP_v1, 5 s windows (6 per clip)")
for m in ("both", "harm", "perc"):
    print(f"\n== mode {m}: matched score | mismatched floor | margin (matched - mismatched), mean over songs")
    for c in conds: print(f"  {c:15s} {ps(c, m).mean():+.3f} | {ps(c, m + '_mismatch').mean():+.3f} | {(ps(c, m) - ps(c, m + '_mismatch')).mean():+.3f}  (songs margin>0: {int(((ps(c, m) - ps(c, m + '_mismatch')) > 0).sum())}/{len(songs)})")
    for a, b in (("neutral", "original_codec"), ("1980s", "neutral"), ("2010s", "neutral"), ("1960s", "neutral"), ("2010s", "1980s")):
        x = (ps(a, m) - ps(a, m + "_mismatch")) - (ps(b, m) - ps(b, m + "_mismatch")); print(f"  margin {a} - {b}: {x.mean():+.3f} [{int((x > 0).sum())}/{len(songs)}] wilcoxon p={(wilcoxon(x).pvalue if np.any(x != 0) else 1.0):.4f}")
