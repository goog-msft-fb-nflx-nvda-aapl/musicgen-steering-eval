"""Genre/style confound test: LEACE (Belrose et al. 2023) erasure of the Discogs-EffNet style information from frozen MERT-v2 embeddings, then re-probe era (A) or market (B).
Style concept Z = top-k principal components (k = 5, 10, 20, 50) of the 400 Discogs-EffNet style-tag activations of the ORIGINAL MIX (a proxy for genre/style: the manifest has no genre column), fitted on TRAIN; eraser fitted on TRAIN only.
Controls: random orthogonal projections of the same rank (30 draws); check that style predictability (ridge R^2 of the k components from the erased embedding, validation) falls. For B also a joint erasure of style (k=10) and sung-language groups (Whisper LID proxy).
usage: python 58_style_erasure.py A|B -> results/style_erasure_<ds>.json"""
import sys, json, os, numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from concept_erasure import LeaceEraser
from common import ROOT, LABELS, manifest
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]; tags = np.load(os.path.join(ROOT, "emb", f"{ds}_dgtags.npy")).astype(np.float64)
def ci(ok, B=2000): bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(B)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def probe(a, b, C): m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(a, y[tr]); ok = (m.predict(b) == y[va]); return float(ok.mean()), ci(ok.astype(float))
def eraser(Xtr, Z): er = LeaceEraser.fit(torch.tensor(Xtr, dtype=torch.float64), torch.tensor(Z, dtype=torch.float64)); return lambda X: er(torch.tensor(X, dtype=torch.float64)).numpy().astype(np.float32)
langs = json.load(open(os.path.join(ROOT, "emb", "lid_langs.json"))); top = np.array([langs[i] for i in np.load(os.path.join(ROOT, "emb", f"{ds}_lid_voc.npy")).argmax(1)])
keep = ["en", "pt", "es", "it", "de"] if ds == "B" else ["en", "es"]; grp = np.array([keep.index(t) if t in keep else len(keep) for t in top]); Kl = len(keep) + 1
out = {"ds": ds, "conds": {}}
for c in ("mix_norm", "voc_norm", "acc_norm"):
    L, C = P[c]["layer"], P[c]["C"]; X = np.load(os.path.join(ROOT, "emb", f"{ds}_{c}.npy"))[:, L].astype(np.float32); Xtr, Xva = X[tr], X[va]; r = {"layer": L}; r["acc_before"], r["ci_before"] = probe(Xtr, Xva, C); r["by_k"] = {}
    pca = PCA(50, random_state=0).fit(tags[tr]); Ztr_all, Zva_all = pca.transform(tags[tr]), pca.transform(tags[va])
    for k in (5, 10, 20, 50):
        f = eraser(Xtr, Ztr_all[:, :k]); Etr, Eva = f(Xtr), f(Xva); a, cc = probe(Etr, Eva, C)
        rg = Ridge(alpha=10.0).fit(StandardScaler().fit(Xtr).transform(Xtr), Ztr_all[:, :k]); r2_before = float(rg.score(StandardScaler().fit(Xtr).transform(Xva), Zva_all[:, :k]))
        sc = StandardScaler().fit(Etr); rg2 = Ridge(alpha=10.0).fit(sc.transform(Etr), Ztr_all[:, :k]); r2_after = float(rg2.score(sc.transform(Eva), Zva_all[:, :k]))
        ra = []
        for _ in range(30): Q, _ = np.linalg.qr(rng.standard_normal((Xtr.shape[1], k))); ra.append(probe(Xtr - Xtr @ Q @ Q.T, Xva - Xva @ Q @ Q.T, C)[0])
        r["by_k"][k] = {"acc_after_style_erasure": a, "ci95": cc, "random_same_rank": {"mean": float(np.mean(ra)), "min": float(np.min(ra)), "max": float(np.max(ra))}, "style_pc_ridge_R2_val_before": r2_before, "style_pc_ridge_R2_val_after": r2_after}
        print(c, k, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r["by_k"][k].items() if kk != "random_same_rank"}), "random", {kk: round(v, 3) for kk, v in r["by_k"][k]["random_same_rank"].items()}, flush=True)
    if ds == "B":
        Z = np.hstack([Ztr_all[:, :10], np.eye(Kl)[grp[tr]]]); f = eraser(Xtr, Z); r["acc_after_style10_plus_language"], r["ci_joint"] = probe(f(Xtr), f(Xva), C); print(c, "joint style10+language", round(r["acc_after_style10_plus_language"], 3), flush=True)
    r["acc_before_print"] = round(r["acc_before"], 3); out["conds"][c] = r
json.dump(out, open(os.path.join(ROOT, "results", f"style_erasure_{ds}.json"), "w"), indent=1)
