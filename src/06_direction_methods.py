"""alternative direction-discovery methods on the cached hidden states, same task/split as 01 (note vs rest, balanced, 70/15/15).
split reproduces 01's python-random call order (random.seed(42); random.sample(pos), random.sample(neg); slice; shuffle train/val/test)
on index lists; random.sample depends only on population size, so this should select the same clips as 01. NOT verified clip-by-clip
(only the split sizes 1158/248/250 are checked).
methods (all give a unit direction oriented towards the positive class):
  meandiff : mu_pos - mu_neg (difference of class means; CAA-style)
  logistic : L2 logistic regression weight vector on standardised features, mapped back to raw space (C=1)
  lda      : shrinkage LDA, Sigma^-1 (mu_pos - mu_neg)
per layer: val auc (selects layers, same role as musicrfm's val score), test acc/auc of the 1-d projection with a threshold at the
midpoint of the train class-mean projections. outputs: directions_alt/<method>/<model>/note_<N>.pkl ({layer: {0: tensor}}),
results_alt/<method>/<model>/note_<N>.json (layer_metrics in the same {layer: {train_results: {score}}} layout as 01 outputs).
usage: python 06_direction_methods.py --model small   (cpu)
"""
import argparse, json, os, pickle, random
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="small"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--suffix", default="", help="output dirs directions_alt<suffix>/results_alt<suffix> (use _s43 etc. for other seeds)")
a = ap.parse_args()
Z = np.load(f"hidden_{a.model}.npz"); H = Z["H"].astype(np.float32); layers = [int(l) for l in Z["layers"]]; names = list(Z["root_note_name"])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def split(note):
    pos = [i for i, x in enumerate(names) if x == note]; neg = [i for i, x in enumerate(names) if x != note]
    random.seed(a.seed); n = min(len(pos), len(neg)); pos, neg = random.sample(pos, n), random.sample(neg, n)
    i1, i2 = int(0.7 * n), int(0.85 * n)
    tr, va, te = pos[:i1] + neg[:i1], pos[i1:i2] + neg[i1:i2], pos[i2:] + neg[i2:]
    for s in (tr, va, te): random.shuffle(s)
    return tr, va, te

def fit(method, X, y):
    if method == "meandiff": d = X[y == 1].mean(0) - X[y == 0].mean(0)
    elif method == "logistic":
        mu, sd = X.mean(0), X.std(0) + 1e-6
        d = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y).coef_[0] / sd
    elif method == "lda": d = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y).coef_[0]
    return d / (np.linalg.norm(d) + 1e-12)

for method in ["meandiff", "logistic", "lda"]:
    for note in NOTES:
        tr, va, te = split(note)
        assert (len(tr), len(va), len(te)) == (1158, 248, 250) or note != "D#", (len(tr), len(va), len(te))
        y = lambda ix: np.array([1 if names[i] == note else 0 for i in ix])
        dirs, lm = {}, {}
        for li, l in enumerate(layers):
            X = lambda ix: H[ix, li]
            d = fit(method, X(tr), y(tr)); dirs[l] = {0: torch.tensor(d)}
            p_tr = X(tr) @ d; thr = 0.5 * (p_tr[y(tr) == 1].mean() + p_tr[y(tr) == 0].mean())
            auc_va = roc_auc_score(y(va), X(va) @ d); p_te = X(te) @ d
            lm[str(l)] = dict(train_results=dict(score=float(auc_va)), test_acc=float(((p_te > thr) == y(te)).mean() * 100), test_auc=float(roc_auc_score(y(te), p_te)))
        for root, obj, dump in ((f"directions_alt{a.suffix}", dirs, lambda o, f: pickle.dump(o, f)), (f"results_alt{a.suffix}", dict(layer_metrics=lm, method=method, note=note, model=a.model, n=[len(tr), len(va), len(te)]), lambda o, f: f.write(json.dumps(o, indent=1).encode()))):
            os.makedirs(f"{root}/{method}/{a.model}", exist_ok=True)
            with open(f"{root}/{method}/{a.model}/note_{note.replace('#', 'sharp')}.{'pkl' if root.startswith('directions_alt') else 'json'}", "wb") as f: dump(obj, f)
        best = max(lm.items(), key=lambda kv: kv[1]["train_results"]["score"])
        print(method, note, "best val-auc layer", best[0], round(best[1]["train_results"]["score"], 3), "test acc there", round(best[1]["test_acc"], 1), flush=True)
