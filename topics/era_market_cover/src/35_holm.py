"""Holm-Bonferroni correction of the paired McNemar tests between stem conditions. Family = the 6 pairwise tests among {mix_norm, voc_norm, acc_norm, recomb_norm} for one dataset and one encoder (probe json 'mcnemar').
usage: python 35_holm.py > results/holm_summary.txt"""
import json, itertools
C = ["mix_norm", "voc_norm", "acc_norm", "recomb_norm"]
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
for ds in "AB":
    for enc, f in (("MERT", f"results/probe_{ds}.json"), ("MuQ", f"results/probe_muq_{ds}.json")):
        m = json.load(open(f))["mcnemar"]; keys, ps = [], []
        for a, b in itertools.combinations(C, 2):
            k = f"{a} vs {b}" if f"{a} vs {b}" in m else f"{b} vs {a}"; keys.append((a, b, m[k])); ps.append(m[k]["p"])
        adj = holm(ps); print(f"== dataset {ds}, {enc}: raw p -> Holm-adjusted p (6 tests)")
        for (a, b, v), p, q in zip(keys, ps, adj): print(f"  {a:12s} vs {b:12s} discordant {v['only_first']}/{v['only_second']}  p={p:.4f} -> {q:.4f}{'  *' if q < 0.05 else ''}")
