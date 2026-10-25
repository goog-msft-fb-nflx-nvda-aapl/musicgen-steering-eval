import json, sys, numpy as np
from scipy.stats import wilcoxon, binomtest
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/content_metrics.json")); songs = sorted({r["clip"] for r in d})
print(f"{len(d)} generations, {len(songs)} songs; chance: top-1 {1/12:.3f}, MRR {np.mean([1/k for k in range(1,13)]):.3f}")
for c in ("neutral", "1960s", "1980s", "2010s", "all"):
    rr = [r for r in d if c == "all" or r["cond"] == c]
    t1 = np.mean([r["top1"] for r in rr]); k = sum(r["top1"] for r in rr)
    print(f"{c:8s} n={len(rr):3d} top-1 {t1:.3f} (binom vs 1/12 p={binomtest(k, len(rr), 1/12, alternative='greater').pvalue:.1e})  MRR {np.mean([r['rr'] for r in rr]):.3f}  chord agree {np.mean([r['chord_agree'] for r in rr]):.3f} vs rotation floor {np.mean([r['chord_rotation_floor'] for r in rr]):.3f}  dBPM {np.mean([r['dbpm'] for r in rr]):.1f}  sim_own {np.mean([r['sim_own'] for r in rr]):.3f}")
g = np.array([r["chord_agree"] - r["chord_rotation_floor"] for r in d]); print(f"chord lift over rotation floor: mean {g.mean():+.3f}, positive in {(g>0).sum()}/{len(g)}, wilcoxon p={wilcoxon(g).pvalue:.1e}")
def per_song(c, k): return np.array([np.mean([r[k] for r in d if r["clip"] == s and r["cond"] == c]) for s in songs])
for k in ("rr", "chord_agree", "dbpm"):
    for e in ("1960s", "1980s", "2010s"):
        x, y = per_song(e, k), per_song("neutral", k); p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        print(f"  {k:12s} {e} - neutral: {np.mean(x - y):+.3f} (songs higher {int((x>y).sum())}/12, wilcoxon p={p:.3f})")
