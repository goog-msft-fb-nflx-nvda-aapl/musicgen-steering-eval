"""Discogs-EffNet decade-probe Delta (same statistic as 56_summarize_gens.py: per-song P(target decade | era prompt) - P(target decade | neutral), mean over seeds; Wilcoxon over the 24 songs) and the three tags with the largest mean activation increase over neutral, for every results/discogs_*.json. Writes results/discogs_table.txt.
The probe (logistic regression on the 1280-d Discogs-EffNet embedding of EnCodec-resynthesized real accompaniment, validation accuracy in the json) is the same for all sets; it is an independent encoder family from MERT/MuQ.
usage: python 82_tagger_table.py"""
import glob, json, os, numpy as np
from scipy.stats import wilcoxon
TD = {"1960s": 0, "1980s": 2, "2010s": 5}
L = ["set | decade probe val acc | Delta P(target decade) 1960s / 1980s / 2010s (songs positive, Wilcoxon p) | top tags vs neutral (1960s; 1980s; 2010s)"]
for f in sorted(glob.glob("results/discogs_*.json")):
    name = os.path.basename(f)[8:-5]
    if name == "vi_meta" or name.startswith("trix_"): continue   # extra-decade sets: see 82b_tagger_six.py
    d = json.load(open(f)); R = d["rows"]; songs = sorted({r["clip"] for r in R}); cells = []
    for e, t in TD.items():
        P = lambda c: np.array([np.mean([r["probs"][t] for r in R if r["clip"] == x and r["cond"] == c]) for x in songs])
        if not any(r["cond"] == e for r in R): cells.append("-"); continue
        dl = P(e) - P("neutral"); p = wilcoxon(dl).pvalue; cells.append(f"{dl.mean():+.3f} [{int((dl > 0).sum())}/{len(songs)}] p={p:.4f}")
    tt = d.get("top_tags_vs_neutral", {}); tags = "; ".join(", ".join(f"{n.split('---')[-1]} {v:+.3f}" for n, v in tt.get(e, [])[:3]) for e in TD)
    L.append(f"{name} | {d['probe_val_acc']:.3f} | " + " / ".join(cells) + " | " + tags)
open("results/discogs_table.txt", "w").write("\n".join(L)); print("\n".join(L))
