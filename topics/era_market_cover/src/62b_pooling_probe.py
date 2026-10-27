"""Pooling ablation, step 2 (CPU): per-condition probe (StandardScaler + logistic regression, C=0.01) for pooling variants of the same MERT-v2 layers: mean (existing emb/), std, max, mean+std, mean+max, three 10 s segment means concatenated (3x1024), per-layer choice by 3-fold CV on TRAIN,
validation accuracy with bootstrap 95% CI (2000 resamples). usage: python 62b_pooling_probe.py A|B -> results/pooling_<ds>.json"""
import sys, os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from common import ROOT, LABELS, manifest
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
L = lambda p: np.load(os.path.join(ROOT, p)).astype(np.float32); out = {"ds": ds, "conds": {}}
for cond in ("mix_norm", "voc_norm", "acc_norm"):
    mean = L(f"emb/{ds}_{cond}.npy"); std = L(f"emb_pool/{ds}_{cond}_std.npy"); mx = L(f"emb_pool/{ds}_{cond}_max.npy"); seg = L(f"emb_pool/{ds}_{cond}_seg3.npy")      # seg: (N,3,25,1024)
    V = {"mean": lambda l: mean[:, l], "std": lambda l: std[:, l], "max": lambda l: mx[:, l], "mean+std": lambda l: np.hstack([mean[:, l], std[:, l]]), "mean+max": lambda l: np.hstack([mean[:, l], mx[:, l]]),
         "seg3_concat": lambda l: seg[:, :, l].reshape(len(seg), -1)}
    out["conds"][cond] = {}
    for name, f in V.items():
        best = (-1, None)
        for l in range(mean.shape[1]):
            X = f(l); s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)), X[tr], y[tr], cv=StratifiedKFold(3, shuffle=True, random_state=0)).mean()
            if s > best[0]: best = (s, l)
        l = best[1]; X = f(l); m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X[tr], y[tr]); ok = (m.predict(X[va]) == y[va]).astype(float); bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]
        out["conds"][cond][name] = {"layer": int(l), "cv": float(best[0]), "val_acc": float(ok.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "dim": int(X.shape[1])}
        print(ds, cond, name, json.dumps(out["conds"][cond][name]), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", f"pooling_{ds}.json"), "w"), indent=1)
