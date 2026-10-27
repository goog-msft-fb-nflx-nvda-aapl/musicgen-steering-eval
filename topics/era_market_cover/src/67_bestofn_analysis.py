"""Best-of-N analysis (G09). For every (song, era in 1960s/1980s/2010s) there are N candidate generations (seeds 10..10+N-1). Selectors: random (= mean over the N candidates, the expected value of one sample), the LALM judge
(P(target decade), multiple-choice readout of 61_lalm_era.py, prior-corrected), identity (chroma-DTW similarity to the original, 13_content_metrics 'sim_own'), judge + identity (sum of within-song z-scores), and the oracle that selects by the MERT era probe itself (headroom, circular).
Outcome metrics of the selected candidate: P(target decade) from the codec-matched MERT and MuQ probes (not used by the judge selectors), judge P(target), identity sim_own, song-id top-1. Paired over songs: selected minus mean of candidates, mean [songs>0], Wilcoxon.
usage: python 67_bestofn_analysis.py --eval results/bon_eval.json --content results/bon_content.json --lalm results/bon_lalm.json > results/bon_summary.txt"""
import argparse, json, numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
ap = argparse.ArgumentParser(); ap.add_argument("--eval", required=True); ap.add_argument("--content", required=True); ap.add_argument("--lalm", required=True); a = ap.parse_args()
TD = {"1960s": 0, "1980s": 2, "2010s": 5}
E = defaultdict(dict)
for r in json.load(open(a.eval))["rows"]:
    if r["cond"] in TD: E[(r["clip"], r["cond"])][r["seed"]] = {"mert": r["mert_probs"][TD[r["cond"]]], "muq": r["muq_probs"][TD[r["cond"]]]}
for c in json.load(open(a.content)):
    if c["cond"] in TD: E[(c["clip"], c["cond"])].setdefault(c["seed"], {}).update({"sim": c["sim_own"], "top1": c["top1"]})
sm = lambda z: (lambda e: e / e.sum())(np.exp(np.array(z) - np.max(z)))
for r in json.load(open(a.lalm))["gens"]["gen_bon"]:
    if r["cond"] in TD: E[(r["clip"], r["cond"])].setdefault(r["seed"], {})["judge"] = float(sm(r["mc"])[TD[r["cond"]]])
z = lambda v: (np.array(v) - np.mean(v)) / (np.std(v) + 1e-9)
out = {}
for era in TD:
    keys = [k for k in E if k[1] == era and all(k2 in E[k] and {"mert", "muq", "sim", "judge"} <= set(E[k][k2]) for k2 in E[k])]
    sel = {"random": [], "judge": [], "identity": [], "judge+identity": [], "oracle_MERT": []}
    for k in keys:
        seeds = sorted(E[k]); J = [E[k][s]["judge"] for s in seeds]; S = [E[k][s]["sim"] for s in seeds]; M = [E[k][s]["mert"] for s in seeds]
        pick = {"judge": int(np.argmax(J)), "identity": int(np.argmax(S)), "judge+identity": int(np.argmax(z(J) + z(S))), "oracle_MERT": int(np.argmax(M))}
        sel["random"].append({m: np.mean([E[k][s][m] for s in seeds]) for m in ("mert", "muq", "judge", "sim", "top1")})
        for name, i in pick.items(): sel[name].append({m: E[k][seeds[i]][m] for m in ("mert", "muq", "judge", "sim", "top1")})
    print(f"== {era}: {len(keys)} songs x N={len(E[keys[0]])} candidates")
    for name in ("judge", "identity", "judge+identity", "oracle_MERT"):
        line = []
        for m, lab in (("mert", "P_MERT"), ("muq", "P_MuQ"), ("judge", "P_judge"), ("sim", "identity sim"), ("top1", "song-id top1")):
            d = np.array([s[m] - r[m] for s, r in zip(sel[name], sel["random"])]); p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            line.append(f"{lab} {d.mean():+.3f} (p={p:.3f})")
        print(f"  selector {name:15s} minus random: " + " | ".join(line))
    print("  random-candidate means:", {m: round(float(np.mean([r[m] for r in sel['random']])), 3) for m in ("mert", "muq", "judge", "sim", "top1")})
