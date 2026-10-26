"""Probe-architecture ablation on the stored mean-pooled embeddings (dataset A decade / B market).
Per (encoder, stem): best layer by 3-fold CV on TRAIN. Then: single-layer LR (baseline), top-3-layer concat, stem fusion (voc+acc concat; late fusion of probabilities), encoder fusion (MERT+MuQ), RBF-SVM, MLP.
All hyperparameters fixed or chosen on TRAIN; val accuracy reported. usage: 45_probe_models.py A|B"""
import sys, json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from common import ROOT, manifest, LABELS
ds = sys.argv[1]; rows = manifest(ds)
y = np.array([LABELS[ds].index(r["label"]) if r["label"] in LABELS[ds] else -1 for r in rows]); sp = np.array([r["split"] for r in rows])
tr, va = np.where((sp == "train") & (y >= 0))[0], np.where((sp == "validation") & (y >= 0))[0]
E = lambda enc, cond: np.load(f"{ROOT}/emb/{ds}_{'muq_' if enc == 'muq' else ''}{cond}.npy").astype(np.float32)
lr = lambda C=0.01: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
cv = StratifiedKFold(3, shuffle=True, random_state=0)
def best_layers(X, k=3):
    s = [cross_val_score(lr(), X[tr][:, l], y[tr], cv=cv).mean() for l in range(X.shape[1])]; return list(np.argsort(s)[::-1][:k]), s
acc = lambda m, X: float((m.predict(X[va]) == y[va]).mean())
out, F, P = {}, {}, {}
for enc in ("mert", "muq"):
    for cond in ("mix_norm", "voc_norm", "acc_norm"):
        X = E(enc, cond); bl, _ = best_layers(X); F[(enc, cond)] = (X, bl)
        m = lr().fit(X[tr][:, bl[0]], y[tr]); out[f"{enc}/{cond}/LR best layer {bl[0]}"] = acc(m, X[:, bl[0]]); P[(enc, cond)] = m.predict_proba(X[va][:, bl[0]])
        Xc = np.concatenate([X[:, l] for l in bl], 1); out[f"{enc}/{cond}/LR concat top3 layers {bl}"] = acc(lr(0.003).fit(Xc[tr], y[tr]), Xc)
        Xb = X[:, bl[0]]
        out[f"{enc}/{cond}/SVM-RBF"] = acc(make_pipeline(StandardScaler(), SVC(C=3, gamma="scale")).fit(Xb[tr], y[tr]), Xb)
        out[f"{enc}/{cond}/MLP"] = float(np.mean([acc(make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1.0, max_iter=400, early_stopping=True, random_state=s)).fit(Xb[tr], y[tr]), Xb) for s in range(3)]))
def feat(enc, cond): X, bl = F[(enc, cond)]; return X[:, bl[0]]
for enc in ("mert", "muq"):
    Xf = np.concatenate([feat(enc, "voc_norm"), feat(enc, "acc_norm")], 1); out[f"{enc}/STEM FUSION concat voc+acc LR"] = acc(lr(0.005).fit(Xf[tr], y[tr]), Xf)
    Xf3 = np.concatenate([feat(enc, c) for c in ("mix_norm", "voc_norm", "acc_norm")], 1); out[f"{enc}/STEM FUSION concat mix+voc+acc LR"] = acc(lr(0.003).fit(Xf3[tr], y[tr]), Xf3)
    pv = (P[(enc, "voc_norm")] + P[(enc, "acc_norm")]) / 2; out[f"{enc}/STEM FUSION late avg voc+acc"] = float((pv.argmax(1) == y[va]).mean())
for cond in ("mix_norm", "voc_norm", "acc_norm"):
    Xe = np.concatenate([feat("mert", cond), feat("muq", cond)], 1); out[f"ENC FUSION concat MERT+MuQ/{cond}"] = acc(lr(0.005).fit(Xe[tr], y[tr]), Xe)
Xall = np.concatenate([feat(e, c) for e in ("mert", "muq") for c in ("voc_norm", "acc_norm")], 1); out["ALL concat MERT+MuQ x voc+acc LR"] = acc(lr(0.002).fit(Xall[tr], y[tr]), Xall)
pa = sum(P[(e, c)] for e in ("mert", "muq") for c in ("voc_norm", "acc_norm")); out["ALL late avg (MERT,MuQ x voc,acc)"] = float((pa.argmax(1) == y[va]).mean())
out["n_train"], out["n_val"] = int(len(tr)), int(len(va))
os.makedirs(f"{ROOT}/results", exist_ok=True); json.dump(out, open(f"{ROOT}/results/probe_models_{ds}.json", "w"), indent=1)
for k, v in out.items(): print(f"{k:60s} {v if isinstance(v, int) else round(v, 3)}")
