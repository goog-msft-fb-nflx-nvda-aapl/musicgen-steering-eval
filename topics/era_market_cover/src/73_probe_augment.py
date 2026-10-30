"""Robustness of the codec-matched era probe to the choice of probe TRAINING data (CPU, stored embeddings).
Probes (logistic regression C=0.01, MERT-v2 layer / MuQ layer as in the codec probes) are trained on dataset-A TRAIN accompaniment stems under three training sets:
  codec  = EnCodec-resynthesized stems (main protocol), raw = original stems, union = both (2x clips; every clip appears once as original and once as codec-resynthesized).
Each variant rescores the stored embeddings of generated sets (results/<set>_eval_emb.npz, the layer used by the codec probes) and the Delta statistic of 56_summarize_gens.py is recomputed (per-song Delta P(target decade | era prompt) - P(. | neutral), mean over songs, 95% song-bootstrap CI).
usage: python 73_probe_augment.py set1 set2 ... -> results/probe_augment.json, results/probe_augment.txt"""
import sys, os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, LABELS, manifest
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; rows = manifest("A"); lab = LABELS["A"]
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
PJ = {"mert": json.load(open(f"{ROOT}/results/probe_codec_mert_A.json")), "muq": json.load(open(f"{ROOT}/results/probe_codec_muq_A.json"))}; L = {k: PJ[k]["conds"]["acccodec_norm"]["layer"] for k in PJ}
E = {k: {c: np.load(f"{ROOT}/emb/A_{pre}{c}.npy").astype(np.float32)[:, L[k]] for c in ("acc_norm", "acccodec_norm")} for k, pre in (("mert", ""), ("muq", "muq_"))}
def fit(k, kind):
    if kind == "codec": X, yy = E[k]["acccodec_norm"][tr], y[tr]
    elif kind == "raw": X, yy = E[k]["acc_norm"][tr], y[tr]
    else: X, yy = np.concatenate([E[k]["acc_norm"][tr], E[k]["acccodec_norm"][tr]]), np.concatenate([y[tr], y[tr]])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X, yy)
KINDS = ("codec", "raw", "union"); M = {(k, kind): fit(k, kind) for k in PJ for kind in KINDS}; rng = np.random.default_rng(0); out = {"layers": L, "val_acc": {}, "sets": {}}; txt = []
for k in PJ:
    for kind in KINDS:
        out["val_acc"][f"{k}_{kind}"] = {c: float((M[(k, kind)].predict(E[k][c][va]) == y[va]).mean()) for c in ("acc_norm", "acccodec_norm")}
txt.append("validation accuracy (probe training set -> accuracy on original / codec-resynthesized validation stems):\n" + "\n".join(f"  {k} {kind}: orig {v['acc_norm']:.3f} codec {v['acccodec_norm']:.3f}" for k in PJ for kind in KINDS for v in [out["val_acc"][f"{k}_{kind}"]]))
for s in sys.argv[1:]:
    R = json.load(open(f"{ROOT}/results/{s}_eval.json"))["rows"]; Z = np.load(f"{ROOT}/results/{s}_eval_emb.npz"); names = list(Z["names"]); assert len(names) == len(R) and all(f"{r['clip']}|{r['cond']}|{r['seed']}" == n for r, n in zip(R, names))
    songs = sorted({r["clip"] for r in R}); res = {}
    for k in PJ:
        X = Z[k]; ok = np.isfinite(X).all(1)
        for kind in KINDS:
            pr = np.full((len(X), 6), np.nan); pr[ok] = M[(k, kind)].predict_proba(X[ok]); cells = {}
            for e, t in TD.items():
                P = lambda c: np.array([np.mean([pr[i, t] for i, r in enumerate(R) if r["clip"] == x and r["cond"] == c]) for x in songs]); d = P(e) - P("neutral")
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; cells[e] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pos": int((d > 0).sum()), "n": len(d)}
            res[f"{k}_{kind}"] = cells
    out["sets"][s] = res
    txt.append(s + f" (n_songs={len(songs)})\n" + "\n".join(f"  {key:12s} " + " | ".join(f"{e} {c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]" for e, c in cells.items()) for key, cells in res.items()))
json.dump(out, open(f"{ROOT}/results/probe_augment.json", "w"), indent=1); open(f"{ROOT}/results/probe_augment.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
