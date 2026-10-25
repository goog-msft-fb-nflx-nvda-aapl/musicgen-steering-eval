"""Summaries and paired tests for results/gen_eval.json. clips (12 songs) are the unit: seeds are averaged within song, then Wilcoxon signed-rank across songs."""
import json, sys, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon, spearmanr
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/gen_eval.json")); R = d["rows"]; songs = sorted({r["clip"] for r in R})
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; true = {r["clip"]: ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"].index(r["true"]) for r in R}
def per_song(cond, key):
    return np.array([np.mean([r[key] for r in R if r["clip"] == s and r["cond"] == cond]) for s in songs])
print("layers used", d["layers"], "| songs", len(songs), "| true decades", sorted(true.values()))
for enc in ("mert", "muq"):
    k = f"{enc}_exp_decade"; print(f"\n== era presence, {enc} decade probe on accompaniment (expected decade index 0=1960s..5=2010s; mean over songs)")
    for c in ("identity", "identity_codec", "dsp_old", "dsp_new", "neutral", "1960s", "1980s", "2010s"): print(f"  {c:9s} {per_song(c, k).mean():.2f}")
    print("  identity vs true decade: spearman", round(float(spearmanr(per_song("identity", k), [true[s] for s in songs])[0]), 2), "| identity_codec:", round(float(spearmanr(per_song("identity_codec", k), [true[s] for s in songs])[0]), 2) if any(r["cond"]=="identity_codec" for r in R) else "n/a")
    for a, b in (("2010s", "1960s"), ("1980s", "1960s"), ("2010s", "1980s"), ("1960s", "neutral"), ("2010s", "neutral"), ("dsp_new", "dsp_old"), ("identity_codec", "identity")):
        x, y = per_song(a, k), per_song(b, k); p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        print(f"  {a} - {b}: mean diff {np.mean(x - y):+.2f}, songs with a>b {int((x > y).sum())}/{len(songs)}, wilcoxon p={p:.4f}")
    pk = f"{enc}_probs"; print("  probability mass on target decade (prompt era / identity-like baselines):")
    for c, t in TD.items(): print(f"   prompt {c}: {np.mean([r[pk][t] for r in R if r['cond']==c]):.3f}  (neutral prompt mass on {c}: {np.mean([r[pk][t] for r in R if r['cond']=='neutral']):.3f}; chance 0.167)")
    print("  top-1 predicted decade equals prompt decade:", {c: round(float(np.mean([int(np.argmax(r[pk])) == t for r in R if r["cond"] == c])), 2) for c, t in TD.items()})
print("\n== melody/harmony preservation (chroma cosine, original mix vs generated remix; control = other song's remix)")
for c in ("neutral", "1960s", "1980s", "2010s"):
    rr = [r for r in R if r["cond"] == c]
    print(f"  {c:8s} frame-aligned {np.mean([r['chroma_cos'] for r in rr]):.3f} (other-song {np.mean([r['chroma_cos_other_song'] for r in rr]):.3f})   dtw {np.mean([r['chroma_dtw'] for r in rr]):.3f} (other-song {np.mean([r['chroma_dtw_other_song'] for r in rr]):.3f})")
a_ = np.array([r["chroma_dtw"] - r["chroma_dtw_other_song"] for r in R if "chroma_dtw" in r]); print(f"  dtw own - other: mean {a_.mean():+.3f}, clips with own>other {(a_>0).sum()}/{len(a_)}, wilcoxon p={wilcoxon(a_).pvalue:.2g}")
a_ = np.array([r["chroma_cos"] - r["chroma_cos_other_song"] for r in R if "chroma_cos" in r]); print(f"  frame-aligned own - other: mean {a_.mean():+.3f}, own>other {(a_>0).sum()}/{len(a_)}, wilcoxon p={wilcoxon(a_).pvalue:.2g}")
