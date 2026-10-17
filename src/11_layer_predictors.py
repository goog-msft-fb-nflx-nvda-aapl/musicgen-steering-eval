"""cheap per-layer predictors of single-layer steering effect, from the cached mean-pooled hidden states (05_cache_hidden.py) and the stage-1 sweep (09_layer_sweep.py).
per (note, layer), meandiff direction d (unit): causal score = dev-prompt hit gain of single-layer steering (mean over eta .6/1.5);
candidate predictors: depth (layer id), probe val auc, hidden norm (mean ||h|| of the pooled states), projection std of the pooled states on d,
raw class-mean separation |mu_pos-mu_neg| (before unit-normalising) and the same divided by hidden norm / projection std ("separation in sd units").
reports pooled spearman (276 points) and mean within-note spearman. cpu only. note: hidden states are time-mean-pooled clips of syntheory audio, not the per-token states seen during generation.
usage: python 11_layer_predictors.py layer_sweep_dir out.json   (run in ~/musicgen_steer on the server, needs hidden_small.npz, directions_alt/meandiff)
"""
import sys, glob, json, pickle, random, numpy as np
from scipy.stats import spearmanr
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
Z = np.load("hidden_small.npz"); H = Z["H"].astype(np.float32); layers = [int(l) for l in Z["layers"]]; names = list(Z["root_note_name"])
rows = []
for n in NOTES:
    sw = json.load(open(f"{sys.argv[1]}/musicgen_small_{n.replace('#', 'sharp')}_meandiff_sweep.json"))
    dirs = pickle.load(open(f"directions_alt/meandiff/small/note_{n.replace('#', 'sharp')}.pkl", "rb"))
    pos = [i for i, x in enumerate(names) if x == n]; neg = [i for i, x in enumerate(names) if x != n]
    random.seed(42); k = min(len(pos), len(neg)); pos, neg = random.sample(pos, k), random.sample(neg, k); tr = pos[:int(.7 * k)] + neg[:int(.7 * k)]
    y = np.array([1] * int(.7 * k) + [0] * int(.7 * k))
    for li, l in enumerate(layers):
        X = H[tr, li]; d = dirs[l][0].numpy(); p = X @ d; raw = X[y == 1].mean(0) - X[y == 0].mean(0)
        rows.append(dict(note=n, layer=l, causal=sw["causal"][str(l)], probe=sw["probe_val_score"][str(l)], depth=-l, hnorm=float(np.linalg.norm(X, axis=1).mean()),
                         pstd=float(p.std()), sep=float(np.linalg.norm(raw)), sep_over_hnorm=float(np.linalg.norm(raw) / np.linalg.norm(X, axis=1).mean()), sep_over_pstd=float(np.linalg.norm(raw) / (p.std() + 1e-9)),
                         pstd_over_hnorm=float(p.std() / np.linalg.norm(X, axis=1).mean())))
out = {}
for key in ["depth", "probe", "hnorm", "pstd", "sep", "sep_over_hnorm", "sep_over_pstd", "pstd_over_hnorm"]:
    pooled = spearmanr([r[key] for r in rows], [r["causal"] for r in rows])[0]
    within = [spearmanr([r[key] for r in rows if r["note"] == n], [r["causal"] for r in rows if r["note"] == n])[0] for n in NOTES]
    out[key] = dict(pooled=float(pooled), within_mean=float(np.nanmean(within)), within_min=float(np.nanmin(within)), within_max=float(np.nanmax(within)))
    print(f"{key:16s} pooled rho {pooled:+.3f}  within-note mean {np.nanmean(within):+.3f} (min {np.nanmin(within):+.2f} max {np.nanmax(within):+.2f})")
json.dump(dict(summary=out, rows=rows), open(sys.argv[2], "w"))
