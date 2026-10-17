"""Q3 analysis of layer_sweep/*.json (09_layer_sweep.py): per note spearman between probe val score and single-layer causal score (dev prompts),
overlap of probe8/causal8, held-out hit by layer set and eta (pooled over notes, per-note paired wilcoxon), and mean causal score by layer depth.
usage: python analysis_layers.py <dir> [method]"""
import sys, glob, json, numpy as np
from scipy.stats import spearmanr, wilcoxon
d = sys.argv[1]; method = sys.argv[2] if len(sys.argv) > 2 else "meandiff"
R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*_{method}_sweep.json"))]
print(len(R), "notes", method)
rho = []
for r in R:
    L = sorted(r["causal"], key=int); rho.append(spearmanr([r["probe_val_score"][l] for l in L], [r["causal"][l] for l in L])[0])
    print(f"{r['note']:3s} rho {rho[-1]:+.2f} overlap {len(set(r['sets']['probe8']) & set(r['sets']['causal8']))}/8 base_dev {r['base_dev']:.2f} base_test {r['base_test']:.2f}")
print("spearman probe-vs-causal: mean", round(float(np.mean(rho)), 3), "range", round(min(rho), 2), round(max(rho), 2))
print("mean overlap probe8/causal8", np.mean([len(set(r["sets"]["probe8"]) & set(r["sets"]["causal8"])) for r in R]))
etas = sorted(R[0]["test_hits"]["probe8"], key=float)
print("held-out hit (prompts 10-19 x2, 12 notes pooled):", "base", round(float(np.mean([r["base_test"] for r in R])), 3))
for e in etas:
    row = {k: np.array([r["test_hits"][k][e] for r in R]) for k in ("probe8", "causal8", "rand8")}
    s = " ".join(f"{k} {v.mean():.3f}" for k, v in row.items())
    p = {}
    for a, b in (("causal8", "probe8"), ("probe8", "rand8"), ("causal8", "rand8")):
        try: p[f"{a}>{b}"] = wilcoxon(row[a], row[b]).pvalue
        except ValueError: p[f"{a}>{b}"] = 1.0
    print(f"eta {e}: {s} | wilcoxon p " + " ".join(f"{k} {v:.3f}" for k, v in p.items()))
L = sorted({int(l) for r in R for l in r["causal"]})
print("mean causal score (single-layer hit gain, dev) and mean probe val score by layer:")
for l in L:
    print(f"{l:4d} causal {np.mean([r['causal'][str(l)] for r in R]):+.3f} probe {np.mean([r['probe_val_score'][str(l)] for r in R]):.3f}")

# fixed-layer controls (09_layer_sweep.py --stage2_only -> *_sets.json): probe8/causal8/rand8/last8/first8 on held-out prompts
S = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*_{method}_sets.json"))]
if S:
    print("\nstage2_only sets,", len(S), "notes, held-out hit pooled; wilcoxon on per-note values")
    for e in sorted(S[0]["test_hits"]["probe8"], key=float):
        row = {k: np.array([r["test_hits"][k][e] for r in S]) for k in S[0]["test_hits"]}
        print(f"eta {e}: " + " ".join(f"{k} {v.mean():.3f}" for k, v in row.items()))
        for a, b in (("causal8", "last8"), ("causal8", "probe8"), ("last8", "probe8"), ("probe8", "rand8")):
            try: p = wilcoxon(row[a], row[b]).pvalue
            except ValueError: p = 1.0
            print(f"    {a} vs {b}: p {p:.3f}", end="")
        print()
