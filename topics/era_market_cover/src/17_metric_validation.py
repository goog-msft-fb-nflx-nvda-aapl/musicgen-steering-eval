"""Era-metric validation and primary statistics from results/gen_eval_codec.json and results/controls_*.json  (SURVEY_metrics.md sections 4 and 5).
 Delta_X per song = P(decade X | prompt X) - P(decade X | neutral prompt), seeds averaged, bootstrap CI over songs, exact Wilcoxon over songs.
 seed ICC(1,1) of the expected decade across the 2 seeds of each (song, prompt); sensitivity curve (crossfade) Spearman; noise and label-swap controls.
 usage: python 17_metric_validation.py results/gen_eval_codec.json results/controls_12.json"""
import json, sys, numpy as np
from scipy.stats import wilcoxon, spearmanr
g = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2])); R = g["rows"]; songs = sorted({r["clip"] for r in R}); TD = {"1960s": 0, "1980s": 2, "2010s": 5}; rng = np.random.default_rng(0)
def boot(x): b = rng.choice(x, size=(5000, len(x))).mean(1); return np.quantile(b, [.025, .975])
print("== Delta_X = P(X | prompt X) - P(X | neutral), mean over songs [95% bootstrap CI over songs], songs with Delta>0, exact Wilcoxon")
for enc in ("mert", "muq"):
    for e, t in TD.items():
        d = np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == s and r["cond"] == e]) - np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == s and r["cond"] == "neutral"]) for s in songs])
        lo, hi = boot(d); print(f"  {enc} {e}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  >0 in {(d>0).sum()}/{len(d)}  p={wilcoxon(d).pvalue:.4f}")
print("\n== seed ICC(1,1) of the expected decade over the 2 seeds of each (song, prompt) (groups =", len(songs) * 4, ")")
for enc in ("mert", "muq"):
    X = np.array([[r[f"{enc}_exp_decade"] for r in R if r["clip"] == s and r["cond"] == e and r["seed"] in (0, 1)] for s in songs for e in ("neutral", "1960s", "1980s", "2010s")])
    k = X.shape[1]; msb = k * X.mean(1).var(ddof=1); msw = X.var(1, ddof=1).mean(); print(f"  {enc}: ICC {(msb - msw) / (msb + (k - 1) * msw):.3f}")
    Y = np.array([[r[f"{enc}_probs"][2] for r in R if r["clip"] == s and r["cond"] == "1980s" and r["seed"] in (0, 1)] for s in songs]); msb = k * Y.mean(1).var(ddof=1); msw = Y.var(1, ddof=1).mean(); print(f"        P(1980s | 1980s prompt) ICC {(msb - msw) / (msb + (k - 1) * msw):.3f}")
print("\n== sensitivity: crossfade original(codec) -> generated, P(target decade) vs alpha (mean over songs and seeds), spearman(alpha, P) averaged over (song, prompt)")
al = c["alphas"]
for enc in ("mert", "muq"):
    for e, t in TD.items():
        cur = np.array([[np.mean([c["probs"][enc][f"sens|{s}|{e}|{sd}|{a}"][t] for sd in (0, 1) if f"sens|{s}|{e}|{sd}|{a}" in c["probs"][enc]]) for a in al] for s in songs])
        rho = np.mean([spearmanr(al, cu)[0] for cu in cur if np.ptp(cu) > 0]); print(f"  {enc} {e}: curve {np.round(cur.mean(0), 3).tolist()}  mean per-song spearman {rho:+.2f}")
print("\n== negative controls: noise through the probes (mean prob per decade, chance 0.167)")
for enc in ("mert", "muq"):
    for kind in ("white", "pink"):
        p = np.array([v for n, v in c["probs"][enc].items() if n.startswith(f"noise|{kind}")]); print(f"  {enc} {kind}: {np.round(p.mean(0), 3).tolist()} (argmax decades {np.bincount(p.argmax(1), minlength=6).tolist()})")
