"""Probe sensitivity (CPU, cached MERT-v2 embeddings at the per-condition CV layer of probe_<ds>.json): (A18) validation accuracy over a grid of the logistic-regression inverse regularization C (the main results use C chosen by CV, mostly 0.01);
(A19) learning curve: validation accuracy when only a fraction (10, 25, 50, 75, 100%) of the TRAIN clips is used (stratified random subsets, 10 draws, mean +- sd; fraction 100% is deterministic). usage: python 63_probe_sensitivity.py A|B -> results/probe_sensitivity_<ds>.json"""
import sys, os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from common import ROOT, LABELS, manifest
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]; out = {"ds": ds, "n_train": len(tr), "n_val": len(va), "conds": {}}
def fit(Xa, ya, Xb, C): m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000)).fit(Xa, ya); return float((m.predict(Xb) == y[va]).mean())
for c in ("mix_norm", "voc_norm", "acc_norm"):
    L, C0 = P[c]["layer"], P[c]["C"]; X = np.load(os.path.join(ROOT, "emb", f"{ds}_{c}.npy"))[:, L].astype(np.float32); r = {"layer": L, "C_selected": C0, "C_grid": {}, "learning_curve": {}}
    for C in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0): r["C_grid"][str(C)] = fit(X[tr], y[tr], X[va], C)
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        accs = []
        for d in range(10 if frac < 1 else 1):
            if frac < 1: idx = next(StratifiedShuffleSplit(1, train_size=frac, random_state=d).split(X[tr], y[tr]))[0]
            else: idx = np.arange(len(tr))
            accs.append(fit(X[tr][idx], y[tr][idx], X[va], C0))
        r["learning_curve"][str(frac)] = {"n": int(len(idx)), "mean": float(np.mean(accs)), "sd": float(np.std(accs))}
    out["conds"][c] = r; print(ds, c, "C-grid", {k: round(v, 3) for k, v in r["C_grid"].items()}, "| curve", {k: (v["n"], round(v["mean"], 3), round(v["sd"], 3)) for k, v in r["learning_curve"].items()}, flush=True)
json.dump(out, open(os.path.join(ROOT, "results", f"probe_sensitivity_{ds}.json"), "w"), indent=1)
