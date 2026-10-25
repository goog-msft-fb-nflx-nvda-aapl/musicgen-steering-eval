"""Mac-side analysis of vocal-bleed diagnostics. usage: python 09_bleed_analysis.py <manifest_dir>  (manifest_A.csv, manifest_B.csv with split/label columns; not committed)"""
import csv, json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
L = {"A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"], "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]}
for d in "AB":
    rows = list(csv.DictReader(open(f"{sys.argv[1]}/manifest_{d}.csv"))); sp = np.array([r["split"] for r in rows]); y = np.array([L[d].index(r["label"]) if r["label"] else -1 for r in rows])
    tr, va = np.where(sp == "train")[0], np.where(sp == "validation")[0]; sh = np.load(f"results/emb_small/{d}_vocshare.npy"); lid = np.log(np.load(f"results/emb_small/{d}_lid_acc.npy") + 1e-6)
    pr = json.load(open(f"results/probe_{d}.json")); yv = y[va]
    best = max((cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), lid[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean(), C) for C in (0.01, 0.1, 1, 10))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000)).fit(lid[tr], y[tr])
    print(f"== {d}: vocal share of energy mean {sh.mean():.2f} (min {sh.min():.2f}, max {sh.max():.2f}); language posterior of ACCOMPANIMENT stem -> label: cv {best[0]:.3f} val {(m.predict(lid[va])==yv).mean():.3f} (chance 0.167)")
    q = np.quantile(sh[va], [1 / 3, 2 / 3]); t = np.digitize(sh[va], q)
    for c in ("mix_norm", "voc_norm", "acc_norm"):
        ok = (np.array(pr["conds"][c]["pred"]) == yv)
        print(f"   {c:9s} val acc by vocal-share tercile (low/mid/high, n={[int((t==i).sum()) for i in range(3)]}): {[round(float(ok[t==i].mean()), 3) for i in range(3)]}  spearman(vocal share, correct)={spearmanr(sh[va], ok)[0]:.2f}")
