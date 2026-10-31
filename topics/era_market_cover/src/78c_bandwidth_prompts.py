"""Era-free bandwidth wording (bw_hi, bw_lo) vs neutral: per-song change of the probe's expected decade E (mean over songs of sum_k k P_k) and of P(each decade), 95% song-bootstrap CI, for MERT and MuQ codec probes; plus the same for the era prompts of a reference set so the sizes can be compared.
usage: python 78c_bandwidth_prompts.py label=eval.json ... -> results/bandwidth_prompts.txt (conditions: every cond except neutral and identity/dsp rows)"""
import sys, json, numpy as np
rng = np.random.default_rng(0); txt = []; D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
for arg in sys.argv[1:]:
    label, ev = arg.split("=", 1); R = json.load(open(ev))["rows"]; songs = sorted({r["clip"] for r in R}); conds = sorted({r["cond"] for r in R if r["cond"] not in ("neutral", "identity", "identity_codec", "dsp_old", "dsp_new")}); txt.append(label)
    for enc in ("mert", "muq"):
        def S(c, f): return np.array([np.mean([f(np.array(r[f"{enc}_probs"])) for r in R if r["clip"] == s and r["cond"] == c]) for s in songs])
        for c in conds:
            def ci(d): b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; return f"{d.mean():+.3f} [{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]"
            dE = S(c, lambda p: p @ np.arange(6)) - S("neutral", lambda p: p @ np.arange(6))
            txt.append(f"  {enc} {c}: dE {ci(dE)} | " + " ".join(f"dP({D[k]}) {ci(S(c, lambda p, k=k: p[k]) - S('neutral', lambda p, k=k: p[k]))}" for k in range(6)))
open("results/bandwidth_prompts.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
