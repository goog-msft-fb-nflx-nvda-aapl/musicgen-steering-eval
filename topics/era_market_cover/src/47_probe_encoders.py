"""Encoder comparison with the 45_probe_models.py protocol: per (encoder, stem) best layer by 3-fold CV on TRAIN, linear probe (C=0.01), val accuracy with bootstrap 95% CI (2000 clip resamples). usage: 47_probe_encoders.py A|B"""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from common import ROOT, manifest, LABELS
ds = sys.argv[1]; rows = manifest(ds)
y = np.array([LABELS[ds].index(r["label"]) if r["label"] in LABELS[ds] else -1 for r in rows]); sp = np.array([r["split"] for r in rows])
tr, va = np.where((sp == "train") & (y >= 0))[0], np.where((sp == "validation") & (y >= 0))[0]
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)); cv = StratifiedKFold(3, shuffle=True, random_state=0)
rng = np.random.default_rng(0); B = rng.integers(0, len(va), (2000, len(va))); out = {}
for enc, pre in (("MERT-v2-30s", ""), ("MuQ-large", "muq_"), ("Whisper-medium enc", "whisper_"), ("AST-AudioSet", "ast_"), ("CLAP-htsat-fused", "clap_")):
    for cond in ("mix_norm", "voc_norm", "acc_norm"):
        try: X = np.load(f"{ROOT}/emb/{ds}_{pre}{cond}.npy").astype(np.float32)
        except FileNotFoundError: continue
        X = X.reshape(len(X), -1, X.shape[-1]); s = [cross_val_score(lr(), X[tr][:, l], y[tr], cv=cv).mean() for l in range(X.shape[1])]; l = int(np.argmax(s))
        c = (lr().fit(X[tr][:, l], y[tr]).predict(X[va][:, l]) == y[va]).astype(float); bs = c[B].mean(1)
        out[f"{enc}/{cond}"] = {"layer": l, "n_layers": X.shape[1], "cv": float(max(s)), "val_acc": float(c.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print(f"{enc:22s} {cond:9s} layer {l:2d}/{X.shape[1]}  cv {max(s):.3f}  val {c.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]", flush=True)
json.dump(out, open(f"{ROOT}/results/probe_encoders_{ds}.json", "w"), indent=1)
