"""Guidance-scale sweep summary (seed 0, 24 songs): era Delta = P(target decade | prompt) - P(target decade | neutral prompt) per song (codec-matched MERT L18 / MuQ L9 probes),
Wilcoxon over songs; content = song-identification top-1 / MRR and chord agreement of the remix against the original. usage: python 21_cfg_analysis.py > results/cfg_summary.txt"""
import json, numpy as np
from scipy.stats import wilcoxon
TD = {"1960s": 0, "1980s": 2, "2010s": 5}
print("config: 24 val songs of dataset A, seed 0, MusicGen-Melody 1.5B, guidance scale g; era probes trained on EnCodec-resynthesized accompaniment (acccodec_norm)")
for enc in ("mert", "muq"):
    print(f"\n== Delta P(target decade), {enc}: mean over songs [songs with Delta>0 / 24], wilcoxon p")
    for g in (1, 3, 5, 7):
        R = json.load(open(f"results/cfg{g}_eval.json"))["rows"]; songs = sorted({r["clip"] for r in R}); s = []
        for e, t in TD.items():
            P = lambda c: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == x and r["cond"] == c]) for x in songs])
            d = P(e) - P("neutral"); p = wilcoxon(d).pvalue; s.append(f"{e} {d.mean():+.3f} [{int((d > 0).sum())}/24] p={p:.4f}")
        print(f"  g={g}: " + " | ".join(s))
print("\n== content preservation (remix vs original mix), mean over 24 songs x 4 prompts")
for g in (1, 3, 5, 7):
    C = json.load(open(f"results/cfg{g}_content.json"))
    print(f"  g={g}: n={len(C)} top1 {np.mean([c['top1'] for c in C]):.3f} MRR {np.mean([c['rr'] for c in C]):.3f} chord agreement {np.mean([c['chord_agree'] for c in C]):.3f} (rotation floor {np.mean([c['chord_rotation_floor'] for c in C]):.3f}) |dBPM| {np.mean([c['dbpm'] for c in C]):.1f}")
