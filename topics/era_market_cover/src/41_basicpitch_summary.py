"""Summary of results/basicpitch.json: onset F1 and note F1 (basic-pitch transcriptions, generated vs original accompaniment) and the mismatched-song floor, per generator and prompt; paired Wilcoxon of own vs other-song floor over 24 songs. usage: python 41_basicpitch_summary.py"""
import json, numpy as np
from scipy.stats import wilcoxon
R = json.load(open("results/basicpitch.json")); songs = sorted({r["clip"] for r in R})
for g in ("gen", "mcl_gen", "gen_textonly"):
    print(f"## {g}")
    for k in ("onset_f1", "note_f1"):
        own = np.array([np.mean([r[k] for r in R if r["gen"] == g and r["clip"] == s]) for s in songs]); oth = np.array([np.mean([r[k + "_other"] for r in R if r["gen"] == g and r["clip"] == s]) for s in songs])
        print(f"  {k}: matched {own.mean():.3f}, other-song floor {oth.mean():.3f}, songs matched>floor {int((own > oth).sum())}/{len(songs)}, wilcoxon p={wilcoxon(own - oth).pvalue:.4f}; by prompt: " + ", ".join(f"{e} {np.mean([r[k] for r in R if r['gen'] == g and r['cond'] == e]):.3f}" for e in ("neutral", "1960s", "1980s", "2010s")))
print("notes per clip (original / generated):", np.mean([r["n_ref"] for r in R]).round(0), {g: float(np.mean([r["n_hyp"] for r in R if r["gen"] == g]).round(0)) for g in ("gen", "mcl_gen", "gen_textonly")})
