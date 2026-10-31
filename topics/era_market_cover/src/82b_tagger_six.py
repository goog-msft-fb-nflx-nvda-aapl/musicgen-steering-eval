"""Discogs-EffNet decade probe on the six-decade sets: joins results/discogs_<main>.json (1960s/1980s/2010s) and results/discogs_<extra>.json (1970s/1990s/2000s) of the same configuration; per-song Delta P(target decade | era prompt) - P(. | neutral) (seed means), Wilcoxon over songs, and mean expected decade index. usage: python 82b_tagger_six.py label=main_tag+extra_tag ... -> results/discogs_six.txt"""
import sys, json, numpy as np
from scipy.stats import wilcoxon
D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; txt = []
for arg in sys.argv[1:]:
    lab, rest = arg.split("=", 1); R = []
    for t in rest.split("+"): R += json.load(open(f"results/discogs_{t}.json"))["rows"]
    songs = sorted({r["clip"] for r in R}); cells = []
    for t, e in enumerate(D):
        P = lambda c: np.array([np.mean([r["probs"][t] for r in R if r["clip"] == s and r["cond"] == c]) for s in songs]); d = P(e) - P("neutral")
        ed = np.mean([np.dot(r["probs"], np.arange(6)) for r in R if r["cond"] == e]); cells.append(f"{e} {d.mean():+.3f} [{int((d > 0).sum())}/{len(songs)}] p={wilcoxon(d).pvalue:.4f} E={ed:.2f}")
    txt.append(f"{lab} (n_songs={len(songs)}): " + " | ".join(cells))
open("results/discogs_six.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
