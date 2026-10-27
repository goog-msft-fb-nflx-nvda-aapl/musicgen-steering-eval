"""Summary of results/lalm_era.json (61_lalm_era.py): positive control on real validation clips and, for each generation directory, the per-song change in the judge's probability of the target decade
(prompt vs neutral prompt), both readouts (multiple choice 'mc', per-decade yes/no 'yn'), prior-corrected scores (yes/no scores are additionally divided by the per-decade standard deviation pooled over all generations before the softmax), Wilcoxon over songs (seeds averaged). usage: python 64_lalm_analysis.py results/lalm_era.json > results/lalm_summary.txt"""
import sys, json, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
d = json.load(open(sys.argv[1])); TD = {"1960s": 0, "1980s": 2, "2010s": 5}
print("positive control, real validation clips (n=132): accuracy / Spearman(expected decade, true) / mean expected decade (true mean 2.5)")
for k in ("real_accomp", "real_mix"):
    print(" ", k, {n: (round(v["acc"], 3), round(v["spearman_expdec_vs_true"], 3), round(v["mean_expdec"], 2)) for n, v in d[k].items()})
sm = lambda Z: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(Z - Z.max(-1, keepdims=True)))
allyn = np.array([r["yn"] for g in d["gens"].values() for r in g]); YN_SD = allyn.std(0)   # fixed per-decade scale from all generations pooled (labels not used)
for g, rows in d["gens"].items():
    out = []
    for key in ("mc", "yn"):
        by = defaultdict(lambda: defaultdict(list))
        for r in rows: by[r["clip"]][r["cond"]].append(sm(np.array(r[key])) if key == "mc" else sm(np.array(r[key]) / YN_SD))
        songs = sorted(by); line = []
        for e, t in TD.items():
            dl = np.array([np.mean([p[t] for p in by[s][e]]) - np.mean([p[t] for p in by[s]["neutral"]]) for s in songs if by[s][e] and by[s]["neutral"]])
            line.append(f"{e} {dl.mean():+.3f} [{int((dl > 0).sum())}/{len(dl)}] p={wilcoxon(dl).pvalue:.4f}")
        neu = np.mean([p @ np.arange(6) for s in songs for p in by[s]["neutral"]]); out.append(f"{key}: " + " | ".join(line) + f" | neutral E[dec] {neu:.2f}")
    print(f"{g} (n={len(rows)})\n  " + "\n  ".join(out))
