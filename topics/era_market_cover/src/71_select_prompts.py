"""G06 selection: for each decade pick the LLM candidate with the largest mean (MERT + MuQ)/2 delta P(target decade | candidate) - P(target decade | neutral) over the 24 TRAIN songs (seed 0).
Also prints the mean over all candidates (what an unselected LLM prompt gives). usage: python 71_select_prompts.py <tag> -> results/g06_<tag>_selected.json, results/g06_<tag>_selection.txt"""
import sys, json, os, numpy as np
from common import ROOT
tag = sys.argv[1]; PJ = json.load(open(os.path.join(ROOT, "results", f"llm_prompts_{tag}.json")))["prompts"]; TD = {"1960s": 0, "1980s": 2, "2010s": 5}; sel, lines = {}, []
D = {}
for k in range(8):
    R = json.load(open(os.path.join(ROOT, "results", f"g06_{tag}_c{k}_eval.json")))["rows"]; songs = sorted({r["clip"] for r in R})
    for dec, t in TD.items():
        v = []
        for enc in ("mert", "muq"):
            P = lambda c: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == x and r["cond"] == c]) for x in songs]); v.append((P(dec) - P("neutral")).mean())
        D[(dec, k)] = v
for dec in TD:
    sc = [np.mean(D[(dec, k)]) for k in range(8)]; b = int(np.argmax(sc)); sel[dec] = PJ[dec][b]
    lines.append(f"{dec}: chosen candidate {b} score {sc[b]:+.3f} (mean over all 8 candidates {np.mean(sc):+.3f}, min {min(sc):+.3f}, max {max(sc):+.3f}); MERT {D[(dec,b)][0]:+.3f} MuQ {D[(dec,b)][1]:+.3f}\n  prompt: {sel[dec]}")
json.dump(sel, open(os.path.join(ROOT, "results", f"g06_{tag}_selected.json"), "w"), indent=1); open(os.path.join(ROOT, "results", f"g06_{tag}_selection.txt"), "w").write("\n".join(lines)); print("\n".join(lines))
