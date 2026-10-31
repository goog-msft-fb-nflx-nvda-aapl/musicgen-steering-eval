"""Prompt-wording variants of one decade (default 1990s, decade index 3): per-song Delta P(target decade | variant) - P(. | neutral), 95% song-bootstrap CI, mean expected decade E, top-1 = target share, and song-ID top-1 / chord agreement from the content json.
usage: python 78b_prompt_variants_decade.py <decade_index> label=eval.json:content.json ... -> results/prompt_variants_decade.txt  (variant conditions are every cond except neutral)"""
import sys, json, numpy as np
t = int(sys.argv[1]); rng = np.random.default_rng(0); txt = []
for arg in sys.argv[2:]:
    label, rest = arg.split("=", 1); ev, ct = rest.split(":"); R = json.load(open(ev))["rows"]; C = json.load(open(ct)); songs = sorted({r["clip"] for r in R}); conds = sorted({r["cond"] for r in R if r["cond"] not in ("neutral", "identity", "identity_codec", "dsp_old", "dsp_new")}); txt.append(label)
    for enc in ("mert", "muq"):
        cells = []
        for c in conds:
            P = lambda x: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == s and r["cond"] == x]) for s in songs]); d = P(c) - P("neutral"); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            E = np.mean([np.dot(r[f"{enc}_probs"], np.arange(6)) for r in R if r["cond"] == c]); top = np.mean([int(np.argmax(r[f"{enc}_probs"]) == t) for r in R if r["cond"] == c])
            cells.append(f"{c} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] E={E:.2f} top1={top:.2f}")
        txt.append(f"  {enc}: " + " | ".join(cells))
    k = "cond" if "cond" in C[0] else "era"; txt.append("  content: " + " | ".join(f"{c} id-top1 {np.mean([x['top1'] for x in C if x[k] == c]):.3f} chord {np.mean([x['chord_agree'] for x in C if x[k] == c]):.3f}" for c in ["neutral"] + conds))
open("results/prompt_variants_decade.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
