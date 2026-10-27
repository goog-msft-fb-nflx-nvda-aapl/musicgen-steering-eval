"""Extra probe controls suggested by survey 3 (all on cached MERT-v2 embeddings, CPU). usage: python 52_extra_controls.py A|B -> results/extra_controls_<ds>.json
(1) dataset cartography on probe training dynamics (Swayamdipta et al. 2020, recalled): a linear softmax probe trained 30 epochs on TRAIN mix_norm; per-clip confidence = mean P(true label) over epochs, variability = std; regions by thresholds below.
(2) INLP (Ravfogel et al. 2020, recalled) removal of sung language (Whisper LID group, a proxy) from mix/voc/acc embeddings: after t iterations (each removes the row space of a language classifier) val accuracy of the target probe vs random same-rank projections.
(3) cross-separator transfer: probe trained on HTDemucs stems evaluated on HTDemucs and on BS-RoFormer stems of the same validation clips (and vice versa)."""
import sys, json, os, numpy as np, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, LABELS, manifest
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0); torch.manual_seed(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]; E = lambda n, L: np.load(os.path.join(ROOT, "emb", f"{ds}_{n}.npy"))[:, L].astype(np.float32)
def ci(ok, B=2000): bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(B)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = {"ds": ds}
# (1) cartography
X = E("mix_norm", P["mix_norm"]["layer"]); sc = StandardScaler().fit(X[tr]); Xt = torch.tensor(sc.transform(X[tr]), dtype=torch.float32); yt = torch.tensor(y[tr])
lin = torch.nn.Linear(Xt.shape[1], len(lab)); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-1); trace = []
for ep in range(30):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 64):
        b = perm[i:i + 64]; opt.zero_grad(); torch.nn.functional.cross_entropy(lin(Xt[b]), yt[b]).backward(); opt.step()
    with torch.no_grad(): trace.append(torch.softmax(lin(Xt), 1)[torch.arange(len(Xt)), yt].numpy())
T = np.array(trace); conf, var = T.mean(0), T.std(0); clow, vhigh = np.percentile(conf, 33), np.percentile(var, 67)
region = np.where(var >= vhigh, "ambiguous", np.where(conf <= clow, "hard", "easy"))
out["cartography"] = {"definition": "ambiguous: variability >= 67th percentile; hard: confidence <= 33rd percentile (and not ambiguous); else easy", "epochs": 30, "mean_conf": float(conf.mean()), "final_train_acc": float((T[-1] > 0.5).mean()), "region_counts": {k: int((region == k).sum()) for k in ("easy", "ambiguous", "hard")},
    "region_by_class": {lab[k]: {r: int(((region == r) & (y[tr] == k)).sum()) for r in ("easy", "ambiguous", "hard")} for k in range(len(lab))}}
gp = os.path.join(ROOT, "results", f"gap_{ds}_clips.csv")
if os.path.exists(gp):
    import csv; G = {r["sample_id"]: int(r["n_wrong"]) for r in csv.DictReader(open(gp))}; ids = [rows[i]["sample_id"] for i in tr]; m = [i for i, s in enumerate(ids) if s in G]
    rho, p = spearmanr(conf[m], [G[ids[i]] for i in m]); out["cartography"]["spearman_conf_vs_oof_n_wrong"] = {"rho": float(rho), "p": float(p), "n": len(m)}
print("cartography", json.dumps(out["cartography"]), flush=True)
# (2) INLP
langs = json.load(open(os.path.join(ROOT, "emb", "lid_langs.json"))); top = np.array([langs[i] for i in np.load(os.path.join(ROOT, "emb", f"{ds}_lid_voc.npy")).argmax(1)])
keep = ["en", "pt", "es", "it", "de"] if ds == "B" else ["en", "es"]; grp = np.array([keep.index(t) if t in keep else len(keep) for t in top]); out["inlp"] = {}
for c in ("mix_norm", "voc_norm", "acc_norm"):
    L, C = P[c]["layer"], P[c]["C"]; Xc = E(c, L); Xtr, Xva = Xc[tr], Xc[va]; mu = Xtr.mean(0); Xtr, Xva = Xtr - mu, Xva - mu
    def acc(a, b, t, v): m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(a, t); return float((m.predict(b) == v).mean())
    rec = [{"iter": 0, "target_acc": acc(Xtr, Xva, y[tr], y[va]), "lang_acc": acc(Xtr, Xva, grp[tr], grp[va]), "rank_removed": 0}]; A, Bv, Q = Xtr.copy(), Xva.copy(), np.zeros((Xtr.shape[1], 0))
    for it in range(1, 9):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(A, grp[tr]); W = m[-1].coef_ / m[0].scale_; U = np.linalg.svd(W, full_matrices=False)[2]; r = int(np.sum(np.linalg.svd(W, compute_uv=False) > 1e-6 * np.linalg.svd(W, compute_uv=False)[0]))
        Qn = U[:r].T; A = A - A @ Qn @ Qn.T; Bv = Bv - Bv @ Qn @ Qn.T; Q = np.hstack([Q, Qn]); rec.append({"iter": it, "target_acc": acc(A, Bv, y[tr], y[va]), "lang_acc": acc(A, Bv, grp[tr], grp[va]), "rank_removed": int(Q.shape[1])})
    rk = rec[-1]["rank_removed"]; ra = []
    for _ in range(10): R, _ = np.linalg.qr(rng.standard_normal((Xtr.shape[1], rk))); ra.append(acc(Xtr - Xtr @ R @ R.T, Xva - Xva @ R @ R.T, y[tr], y[va]))
    out["inlp"][c] = {"trace": rec, "random_same_rank": {"rank": rk, "mean": float(np.mean(ra)), "min": float(np.min(ra)), "max": float(np.max(ra))}, "language_majority_val": float(np.bincount(grp[va]).max() / len(va))}
    print("inlp", c, json.dumps([(r["iter"], round(r["target_acc"], 3), round(r["lang_acc"], 3)) for r in rec]), "random", out["inlp"][c]["random_same_rank"], flush=True)
# (3) cross-separator
out["cross_separator"] = {}
for c in ("voc_norm", "acc_norm"):
    L, C = P[c]["layer"], P[c]["C"]; ht, bs = E(c, L), E("bsr_" + c, L); r = {}
    for a, an, b, bn in ((ht, "htdemucs", bs, "bsr"), (bs, "bsr", ht, "htdemucs")):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(a[tr], y[tr])
        for tgt, tn in ((a, an), (b, bn)): ok = (m.predict(tgt[va]) == y[va]).astype(float); r[f"train_{an}_test_{tn}"] = {"acc": float(ok.mean()), "ci95": ci(ok)}
    out["cross_separator"][c] = r; print("xsep", c, json.dumps({k: round(v["acc"], 3) for k, v in r.items()}), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", f"extra_controls_{ds}.json"), "w"), indent=1)
