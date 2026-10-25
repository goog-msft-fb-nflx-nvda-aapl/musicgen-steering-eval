"""Frozen-embedding probes per condition. Layer and C chosen by 5-fold CV on the train split only, scored once on validation.
usage: python 04_probe.py --ds A --emb ~/era_market/emb --out results/probe_A.json [--conds ...]
validation accuracy comes with a bootstrap 95% CI (2000 resamples of clips); pairs of conditions use exact McNemar on the same clips.
note: splits are artist-disjoint across train/validation, but the CV inside train is not artist-grouped (artist ids are anonymized), so CV is used only to choose layer and C."""
import argparse, json, os, itertools, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import LABELS, manifest

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--emb", default=os.path.expanduser("~/era_market/emb"))
ap.add_argument("--out", required=True); ap.add_argument("--conds", nargs="+", default=["mix_raw", "mix_norm", "voc_raw", "voc_norm", "acc_raw", "acc_norm", "recomb_norm"])
ap.add_argument("--layers", type=int, nargs="*", default=None); ap.add_argument("--prefix", default="", help="embedding file prefix, e.g. muq_"); a = ap.parse_args()
rows = manifest(a.ds); lab = LABELS[a.ds]
split = np.array([r["split"] for r in rows]); y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows])
tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
res = {"ds": a.ds, "n_train": len(tr), "n_val": len(va), "conds": {}}; preds = {}
rng = np.random.default_rng(0); boots = rng.integers(0, len(va), size=(2000, len(va)))
for c in a.conds:
    f = os.path.join(a.emb, f"{a.ds}_{a.prefix}{c}.npy")
    if not os.path.exists(f): continue
    E = np.load(f).astype(np.float32); assert np.isfinite(E).all(), f"non-finite values in {f}"; L = E.shape[1]; layers = a.layers or list(range(L))
    best = (-1, None, None); curve = {}
    for l in layers:
        for C in (0.01, 0.1, 1.0):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            s = cross_val_score(m, E[tr, l], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            curve[f"{l}_{C}"] = float(s)
            if s > best[0]: best = (s, l, C)
    s, l, C = best
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E[tr, l], y[tr])
    p = m.predict(E[va, l]); ok = (p == y[va]).astype(int); preds[c] = ok
    bs = ok[boots].mean(1)
    res["conds"][c] = {"layer": int(l), "C": C, "cv_acc": float(s), "val_acc": float(ok.mean()), "val_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                       "pred": p.tolist(), "layer_cv_best_by_layer": {str(k): max(v for kk, v in curve.items() if kk.split("_")[0] == str(k)) for k in layers}}
    print(a.ds, c, f"layer {l} C {C} cv {s:.3f} val {ok.mean():.3f}", flush=True)
res["y_val"] = y[va].tolist(); res["mcnemar"] = {}
for c1, c2 in itertools.combinations(preds, 2):
    b = int(((preds[c1] == 1) & (preds[c2] == 0)).sum()); d = int(((preds[c1] == 0) & (preds[c2] == 1)).sum())
    res["mcnemar"][f"{c1} vs {c2}"] = {"only_first": b, "only_second": d, "p": float(binomtest(b, b + d, .5).pvalue) if b + d else 1.0}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
