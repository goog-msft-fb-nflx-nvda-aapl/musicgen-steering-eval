"""Song-identification top-1 and chord agreement of the extra-decade (1970s/1990s/2000s) remixes vs the original mix, from results/trix_*_content.json (13_content_metrics.py). usage: python 79_extra_content_table.py tag ... -> results/extra_content.txt"""
import sys, json, numpy as np
ER = ["neutral", "1970s", "1990s", "2000s"]; txt = []
for t in sys.argv[1:]:
    C = json.load(open(f"results/{t}_content.json")); k = "cond" if "cond" in C[0] else "era"
    txt.append(f"{t} (n={len(C)}): " + " | ".join(f"{e} top1 {np.mean([c['top1'] for c in C if c[k] == e]):.3f} chord {np.mean([c['chord_agree'] for c in C if c[k] == e]):.3f}" for e in ER))
open("results/extra_content.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
