"""Summary of results/clap_era.json (zero-shot CLAP decade scores): positive control on real validation clips, and prompt Delta = P_CLAP(target decade | prompt) - P_CLAP(target | neutral) per song, Wilcoxon over 24 songs. usage: python 28_clap_analysis.py [clap_era.json]"""
import json, sys, numpy as np
from scipy.stats import wilcoxon, spearmanr
D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/clap_era.json"))
print("== positive control: real validation clips of dataset A (n=%d), chance 1/6 = 0.167" % len(d["real_val"]))
for k in ("acc", "mix"):
    P = np.array([r[k] for r in d["real_val"]]); y = np.array([D.index(r["true"]) for r in d["real_val"]]); ed = P @ np.arange(6)
    print(f"  {k}: zero-shot top-1 {np.mean(P.argmax(1) == y):.3f}, spearman(expected decade, true decade) {spearmanr(ed, y)[0]:.3f}, mean |expected-true| {np.mean(np.abs(ed - y)):.2f} decades, predicted-decade histogram {np.bincount(P.argmax(1), minlength=6).tolist()}")
G = d["gen"]; songs = sorted({r["clip"] for r in G}); TD = {"1960s": 0, "1980s": 2, "2010s": 5}
def ps(c, t): return np.array([np.mean([r["p"][t] for r in G if r["clip"] == s and r["cond"] == c]) for s in songs])
print("\n== generated accompaniment, mean over seeds, %d songs: Delta P(target decade | prompt) - P(target | neutral)" % len(songs))
for e, t in TD.items():
    x = ps(e, t) - ps("neutral", t); print(f"  {e}: {x.mean():+.3f} [{int((x > 0).sum())}/{len(songs)}] wilcoxon p={wilcoxon(x).pvalue:.4f}   (neutral P={ps('neutral', t).mean():.3f}, prompt P={ps(e, t).mean():.3f})")
ed = lambda c: np.array([np.mean([np.dot(r["p"], np.arange(6)) for r in G if r["clip"] == s and r["cond"] == c]) for s in songs])
print("  expected decade index: " + ", ".join(f"{c} {ed(c).mean():.2f}" for c in ("identity", "identity_codec", "neutral", "1960s", "1980s", "2010s")))
