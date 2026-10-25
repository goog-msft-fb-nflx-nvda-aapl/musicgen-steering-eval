"""Shortcut baselines and nested comparisons, same protocol as 04 (CV on train picks C, scored once on validation).
 - stats_{mix,voc,acc}: 9 signal statistics only
 - lid_{voc,mix}: Whisper language posterior only (100-d)
 - emb+lid: embedding of a condition (layer from probe json) concatenated with vocal-stem language posterior
 McNemar for each against the reference condition on the same validation clips.  usage: python 06_extra_probes.py --ds B --emb ~/era_market/emb --probe results/probe_B.json --out results/extra_B.json"""
import argparse, json, os, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--emb", default=os.path.expanduser("~/era_market/emb"))
ap.add_argument("--probe", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = manifest(a.ds); lab = LABELS[a.ds]; pr = json.load(open(a.probe))
split = np.array([r["split"] for r in rows]); y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows])
tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
L = lambda n: np.load(os.path.join(a.emb, f"{a.ds}_{n}.npy")).astype(np.float32)
def fit(X, Cs=(0.01, 0.1, 1.0, 10.0)):
    best = max(((cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), X[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean(), C) for C in Cs))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000)).fit(X[tr], y[tr]); return (m.predict(X[va]) == y[va]).astype(int), best
feats = {f"stats_{s}": np.clip(np.nan_to_num(L(f"stats_{s}"), neginf=-100, posinf=100), -100, 100000) for s in ("mix", "voc", "acc")}
feats.update({f"lid_{s}": np.log(L(f"lid_{s}") + 1e-6) for s in ("voc", "mix")})
for c in ("mix_norm", "acc_norm", "voc_norm"):
    l = pr["conds"][c]["layer"]; feats[f"{c}_L{l}+lid_voc"] = np.hstack([L(c)[:, l], feats["lid_voc"]])
feats["stats_mix+lid_voc"] = np.hstack([feats["stats_mix"], feats["lid_voc"]])
res = {"ds": a.ds, "conds": {}, "mcnemar_vs": {}}; ok = {}
rng = np.random.default_rng(0); boots = rng.integers(0, len(va), size=(2000, len(va)))
for k, X in feats.items():
    o, best = fit(X); ok[k] = o; bs = o[boots].mean(1)
    res["conds"][k] = {"cv_acc": float(best[0]), "C": best[1], "val_acc": float(o.mean()), "val_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    print(a.ds, k, f"cv {best[0]:.3f} val {o.mean():.3f}", flush=True)
yv = y[va]
for ref in ("mix_norm", "acc_norm", "voc_norm"):
    pref = (np.array(pr["conds"][ref]["pred"]) == yv).astype(int)
    for k, o in ok.items():
        b = int(((o == 1) & (pref == 0)).sum()); d = int(((o == 0) & (pref == 1)).sum())
        res["mcnemar_vs"][f"{k} vs {ref}"] = {"only_first": b, "only_second": d, "p": float(binomtest(b, b + d, .5).pvalue) if b + d else 1.0}
json.dump(res, open(a.out, "w"))
