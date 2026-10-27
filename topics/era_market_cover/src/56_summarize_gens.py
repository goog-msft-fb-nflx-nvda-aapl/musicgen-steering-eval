"""Summarize generator runs with the same statistic as 21_cfg_analysis.py: per-song Delta P(target decade | era prompt) - P(target decade | neutral) from the codec-matched MERT / MuQ probes (mean over songs, [songs > 0], Wilcoxon p),
plus content (song-id top-1, chord agreement). usage: python 56_summarize_gens.py label=eval.json[:content.json][:seeds] ... > results/gens_summary_<tag>.txt   (seeds comma separated, default all)"""
import sys, json, numpy as np
from scipy.stats import wilcoxon
TD = {"1960s": 0, "1980s": 2, "2010s": 5}
for arg in sys.argv[1:]:
    label, rest = arg.split("=", 1); parts = rest.split(":"); ev = parts[0]; ct = parts[1] if len(parts) > 1 and parts[1] else None; seeds = [int(x) for x in parts[2].split(",")] if len(parts) > 2 else None
    R = json.load(open(ev))["rows"]; R = [r for r in R if seeds is None or r["seed"] in seeds]; songs = sorted({r["clip"] for r in R}); line = [f"{label} (n_songs={len(songs)})"]
    for enc in ("mert", "muq"):
        s = []
        for e, t in TD.items():
            P = lambda c: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == x and r["cond"] == c]) for x in songs]); d = P(e) - P("neutral")
            s.append(f"{e} {d.mean():+.3f} [{int((d > 0).sum())}/{len(songs)}] p={wilcoxon(d).pvalue:.4f}")
        line.append(f"  {enc}: " + " | ".join(s))
    if ct:
        C = [c for c in json.load(open(ct)) if seeds is None or c["seed"] in seeds]; line.append(f"  content (n={len(C)}): top1 {np.mean([c['top1'] for c in C]):.3f} chord agreement {np.mean([c['chord_agree'] for c in C]):.3f}")
    print("\n".join(line))
