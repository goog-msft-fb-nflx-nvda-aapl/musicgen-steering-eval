"""Side-by-side summary of the generators on the same 24 songs x 4 prompts x 3 seeds: MusicGen-Melody (gen_eval_codec_24s3, content_metrics_24s3), MuseControlLite (mcl_*), MusicGen-Melody text-only (textonly_*).
Delta = P(target decade | prompt) - P(target | neutral) per song (seeds averaged; codec-matched MERT L22 / MuQ L9 probes), Wilcoxon over songs; CLAP zero-shot Delta; content (song-id top-1, chord agreement); COCOLA margin (matched - mismatched vocal).
usage: python 34_compare_generators.py > results/generators_summary.txt"""
import json, os, numpy as np
from scipy.stats import wilcoxon
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; G = {"MusicGen-Melody (melody+text)": ("gen_eval_codec_24s3", "content_metrics_24s3", "clap_era", "cocola"), "MuseControlLite (melody+text)": ("mcl_eval", "mcl_content", "mcl_clap_era", "mcl_cocola"),
     "MusicGen-Melody text-only": ("textonly_eval", "textonly_content", "textonly_clap_era", None)}
def delta(R, key, t, e):
    songs = sorted({r["clip"] for r in R}); P = lambda c: np.array([np.mean([key(r)[t] for r in R if r["clip"] == s and r["cond"] == c]) for s in songs]); d = P(e) - P("neutral"); return d.mean(), int((d > 0).sum()), len(songs), wilcoxon(d).pvalue if np.any(d != 0) else 1.0
for name, (ev, ct, cl, co) in G.items():
    if not os.path.exists(f"results/{ev}.json"): print(f"\n## {name}: (missing {ev})"); continue
    print(f"\n## {name}"); R = json.load(open(f"results/{ev}.json"))["rows"]
    for enc in ("mert", "muq"):
        print(f"  {enc} Delta: " + " | ".join("%s %+.3f [%d/%d] p=%.4f" % ((e,) + delta(R, lambda r: r[f"{enc}_probs"], t, e)) for e, t in TD.items()))
    if os.path.exists(f"results/{cl}.json"):
        C = json.load(open(f"results/{cl}.json"))["gen"]; print("  CLAP zero-shot Delta: " + " | ".join("%s %+.3f [%d/%d] p=%.4f" % ((e,) + delta(C, lambda r: r["p"], t, e)) for e, t in TD.items()))
    if os.path.exists(f"results/{ct}.json"):
        X = json.load(open(f"results/{ct}.json")); print(f"  content (n={len(X)}): song-id top-1 {np.mean([c['top1'] for c in X]):.3f}, MRR {np.mean([c['rr'] for c in X]):.3f}, chord agreement {np.mean([c['chord_agree'] for c in X]):.3f} (floor {np.mean([c['chord_rotation_floor'] for c in X]):.3f}), |dBPM| {np.mean([c['dbpm'] for c in X]):.1f}")
    if co and os.path.exists(f"results/{co}.json"):
        K = json.load(open(f"results/{co}.json")); songs = sorted({r["clip"] for r in K}); m = lambda c, k: np.mean([np.mean([r[k] for r in K if r["clip"] == s and r["cond"] == c]) for s in songs])
        print("  COCOLA (both) margin matched-mismatched: " + ", ".join(f"{c} {m(c, 'both') - m(c, 'both_mismatch'):+.2f}" for c in ("original_codec", "neutral", "1960s", "1980s", "2010s")))
