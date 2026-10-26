"""Sparse autoencoder on frozen stem embeddings (course idea 5, explainability), CPU/GPU small. For dataset ds and condition cond (layer from the probe json): standardize, train an SAE (2048 latents, TopK-32 activation, no L1; a first ReLU+L1 version with 4096 latents was dense (about 1900 active latents per clip, R^2 about 1) and was discarded) on ALL clips of that dataset (unlabeled; test labels are hidden anyway),
then: (a) reconstruction R^2 and mean active latents per clip; (b) latents ranked by one-way ANOVA F against the label on TRAIN; logistic regression on the top-k latents (k=16, 64, 256) scored on validation, vs the dense-probe accuracy;
(c) number of latents with training-label F above the 99th percentile of F under label permutation (max over latents). usage: python 39_sae.py --out results/sae.json"""
import argparse, json, os, numpy as np, torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; res = {}
for ds, prs in (("A", ("", "muq_")), ("B", ("", "muq_"))):
    rows = manifest(ds); y = np.array([LABELS[ds].index(r["label"]) if r["label"] else -1 for r in rows]); sp = np.array([r["split"] for r in rows]); tr, va = np.where(sp == "train")[0], np.where(sp == "validation")[0]
    for pre in prs:
        pj = json.load(open(os.path.join(ROOT, "results", f"probe_{pre}{ds}.json")))["conds"]
        for c in ("mix_norm", "voc_norm", "acc_norm"):
            l = pj[c]["layer"]; E = np.load(os.path.join(ROOT, "emb", f"{ds}_{pre}{c}.npy")).astype(np.float32)[:, l]; X = (E - E[tr].mean(0)) / (E[tr].std(0) + 1e-6); Xt = torch.from_numpy(X).to(dev); torch.manual_seed(0)
            D, H = X.shape[1], 2048; enc = torch.nn.Linear(D, H).to(dev); dec = torch.nn.Linear(H, D).to(dev); opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
            topk = lambda h: torch.zeros_like(h).scatter_(1, torch.relu(h).topk(32, dim=1).indices, torch.relu(h).topk(32, dim=1).values)
            for ep in range(1500):
                opt.zero_grad(); z = topk(enc(Xt)); loss = ((dec(z) - Xt) ** 2).mean(); loss.backward(); opt.step()
            with torch.no_grad(): z = topk(enc(Xt)); r2 = 1 - ((dec(z) - Xt) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum(); Z = z.cpu().numpy(); act = float((Z > 1e-6).sum(1).mean())
            ok = Z[tr].std(0) > 1e-6; F = np.full(H, 0.0); F[ok] = np.nan_to_num(f_classif(Z[tr][:, ok], y[tr])[0]); order = np.argsort(-F); rng = np.random.default_rng(0)
            null = [np.nan_to_num(f_classif(Z[tr][:, ok], rng.permutation(y[tr]))[0]).max() for _ in range(20)]; r = {"layer": int(l), "r2": float(r2), "active_latents_per_clip": act, "dense_probe_val_acc": pj[c]["val_acc"], "n_latents_F_above_perm_max": int((F > max(null)).sum())}
            for k in (16, 64, 256):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Z[tr][:, order[:k]], y[tr]); r[f"top{k}_val_acc"] = float((m.predict(Z[va][:, order[:k]]) == y[va]).mean())
            res[f"{ds}/{pre or 'mert_'}{c}"] = r; print(ds, pre or "mert_", c, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
