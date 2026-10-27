"""LEACE concept erasure of sung language from frozen MERT-v2 embeddings (Belrose et al. 2023, arXiv 2306.03819), then re-probing era (A) or market (B).
Language label = Whisper LID argmax on the vocal stem (a proxy, not ground truth) grouped into en / pt / es / it / de / other (A: en / es / other). Eraser fitted on TRAIN only, applied to train and validation,
probe re-fitted on erased TRAIN (same layer and C as probe_<ds>.json, StandardScaler + logistic regression). Controls: (i) language probe accuracy before/after erasure (does the eraser work), (ii) 50 random rank-matched
orthogonal projections, (iii) erasing the target label itself (should fall to chance). usage: python 51_leace.py A|B -> results/leace_<ds>.json"""
import sys, json, os, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from concept_erasure import LeaceEraser
from common import ROOT, LABELS, manifest
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
LID = os.environ.get("LIDNAME", "lid"); langs = json.load(open(os.path.join(ROOT, "emb", LID + "_langs.json"))); top = np.array([langs[i] for i in np.load(os.path.join(ROOT, "emb", f"{ds}_{LID}_voc.npy")).argmax(1)])
keep = ["en", "pt", "es", "it", "de"] if ds == "B" else ["en", "es"]; grp = np.array([keep.index(t) if t in keep else len(keep) for t in top]); K = len(keep) + 1
print("language groups", keep + ["other"], np.bincount(grp[tr], minlength=K), flush=True)
P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]
def fitscore(Xtr, ytr, Xva, yva, C):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr, ytr); return float((m.predict(Xva) == yva).mean()), m.predict(Xva)
def ci(pred, yva, B=2000):
    ok = (pred == yva).astype(float); bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(B)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def leace(Xtr, Z): 
    er = LeaceEraser.fit(torch.tensor(Xtr, dtype=torch.float64), torch.tensor(Z, dtype=torch.float64)); return lambda X: er(torch.tensor(X, dtype=torch.float64)).numpy().astype(np.float32), int(er.proj_left.shape[1]) if hasattr(er, "proj_left") else -1
onehot = lambda g, k: np.eye(k)[g]
out = {"ds": ds, "language_groups": keep + ["other"], "train_group_counts": np.bincount(grp[tr], minlength=K).tolist(), "conds": {}}
for c in ("mix_norm", "voc_norm", "acc_norm"):
    L = P[c]["layer"]; C = P[c]["C"]; X = np.load(os.path.join(ROOT, "emb", f"{ds}_{c}.npy"))[:, L].astype(np.float32); Xtr, Xva = X[tr], X[va]; r = {"layer": L, "C": C}
    r["acc_before"], pr0 = fitscore(Xtr, y[tr], Xva, y[va], C); r["ci_before"] = ci(pr0, y[va]); r["lang_acc_before"], _ = fitscore(Xtr, grp[tr], Xva, grp[va], C); r["lang_majority_val"] = float(np.bincount(grp[va], minlength=K).max() / len(va))
    f, rank = leace(Xtr, onehot(grp[tr], K)); Etr, Eva = f(Xtr), f(Xva); r["eraser_rank"] = rank
    r["acc_after_language_erasure"], pr1 = fitscore(Etr, y[tr], Eva, y[va], C); r["ci_after"] = ci(pr1, y[va]); r["lang_acc_after"], _ = fitscore(Etr, grp[tr], Eva, grp[va], C)
    d = Xtr.shape[1]; rk = K - 1; accs = []
    for _ in range(50):
        Q, _ = np.linalg.qr(rng.standard_normal((d, rk))); accs.append(fitscore(Xtr - Xtr @ Q @ Q.T, y[tr], Xva - Xva @ Q @ Q.T, y[va], C)[0])
    r["acc_after_random_rank%d_erasure" % rk] = {"mean": float(np.mean(accs)), "min": float(np.min(accs)), "max": float(np.max(accs))}
    f2, _ = leace(Xtr, onehot(y[tr], len(lab))); r["acc_after_target_label_erasure"], _ = fitscore(f2(Xtr), y[tr], f2(Xva), y[va], C)
    out["conds"][c] = r; print(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", f"leace_{ds}" + ("" if LID == "lid" else "_" + LID) + ".json"), "w"), indent=1)
