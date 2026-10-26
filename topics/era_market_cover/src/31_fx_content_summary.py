"""Summaries: (a) production-only control (results/fx_eval.json, REAL accompaniment through hand-built era chains, no generation): Delta P(target decade | chain) - P(target | unprocessed) per song with the codec-matched probes, next to MusicGen g=3;
(b) content metrics over 24 songs x 4 prompts x 3 seeds (results/content_metrics_24s3.json). usage: python 31_fx_content_summary.py"""
import json, numpy as np
from scipy.stats import wilcoxon
TD = {"1960s": 0, "1980s": 2, "2010s": 5}
for name, f in (("production-only chains on real accompaniment (fx)", "results/fx_eval.json"), ("MusicGen-Melody g=3, seed 0", "results/cfg3_eval.json")):
    R = json.load(open(f))["rows"]; songs = sorted({r["clip"] for r in R}); print(f"== {name}: Delta P(target decade) vs neutral, mean [songs>0 /{len(songs)}] wilcoxon p")
    for enc in ("mert", "muq"):
        s = []
        for e, t in TD.items():
            P = lambda c: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == x and r["cond"] == c]) for x in songs]); d = P(e) - P("neutral")
            s.append(f"{e} {d.mean():+.3f} [{int((d > 0).sum())}] p={wilcoxon(d).pvalue:.4f}")
        print(f"  {enc}: " + " | ".join(s))
C = json.load(open("results/content_metrics_24s3.json")); print(f"\n== content preservation, n={len(C)} remixes (24 songs x 4 prompts x 3 seeds)")
print(f"  song-id top-1 {np.mean([c['top1'] for c in C]):.3f}, MRR {np.mean([c['rr'] for c in C]):.3f}, chord agreement {np.mean([c['chord_agree'] for c in C]):.3f} (rotation floor {np.mean([c['chord_rotation_floor'] for c in C]):.3f}), |dBPM| {np.mean([c['dbpm'] for c in C]):.1f}")
for e in ("neutral", "1960s", "1980s", "2010s"): X = [c for c in C if c["cond"] == e]; print(f"  {e:8s} top-1 {np.mean([c['top1'] for c in X]):.3f} chord {np.mean([c['chord_agree'] for c in X]):.3f}")
