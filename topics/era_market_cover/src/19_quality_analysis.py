"""Audiobox Aesthetics summaries (results/quality*.json): per condition mean of PQ, PC, CE, CU over songs (seeds averaged), paired Wilcoxon of era prompts vs neutral and of generated vs original accompaniment."""
import json, sys, numpy as np
from scipy.stats import wilcoxon
q = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/quality_24.json")); songs = sorted({n.split("|")[0] for n in q if not n.startswith("noise")})
def ps(c, k): return np.array([np.mean([v[k] for n, v in q.items() if n.startswith(s + "|" + c + "|")]) for s in songs])
print(len(songs), "songs; Audiobox Aesthetics axes: PQ production quality, PC production complexity, CE content enjoyment, CU content usefulness (0-10)")
print(f"{'cond':15s}" + "".join(f"{k:>8s}" for k in ("PQ", "PC", "CE", "CU")))
for c in ("identity", "identity_codec", "dsp_old", "dsp_new", "neutral", "1960s", "1980s", "2010s"): print(f"{c:15s}" + "".join(f"{ps(c, k).mean():8.2f}" for k in ("PQ", "PC", "CE", "CU")))
print(f"{'white noise':15s}" + "".join(f"{np.mean([v[k] for n, v in q.items() if n.startswith('noise')]):8.2f}" for k in ("PQ", "PC", "CE", "CU")))
for k in ("PQ", "CE"):
    for e in ("1960s", "1980s", "2010s"):
        d = ps(e, k) - ps("neutral", k); print(f"{k} {e} - neutral: {d.mean():+.2f} (songs higher {int((d>0).sum())}/{len(d)}, wilcoxon p={wilcoxon(d).pvalue:.3f})")
    d = ps("neutral", k) - ps("identity", k); print(f"{k} neutral generation - original accompaniment: {d.mean():+.2f} (songs higher {int((d>0).sum())}/{len(d)}, p={wilcoxon(d).pvalue:.4f})")
    d = ps("identity_codec", k) - ps("identity", k); print(f"{k} codec-resynthesized original - original: {d.mean():+.2f} (p={wilcoxon(d).pvalue:.4f})")
