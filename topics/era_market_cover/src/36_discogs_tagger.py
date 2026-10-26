"""Third, supervised-on-genre evaluator: Essentia Discogs-EffNet (discogs-effnet-bs64-1: 1280-d embedding; genre_discogs400 head: 400 Discogs style tags). Run in venv_ess (CPU).
(1) embeds, at 16 kHz, the codec-resynthesized real accompaniment of all dataset-A clips (probe training/validation), the real accompaniment and generated accompaniment of the evaluation songs (GENDIR, seeds), saves emb_discogs.npz (embedding + tags per item);
(2) decade probe on the embedding (logreg, C by 5-fold CV on train, validation scored once), Delta of the generated accompaniment as in 10_eval_gen; (3) top style tags per prompt (mean tag activation minus neutral).
usage: python 36_discogs_tagger.py --out results/discogs_eval.json"""
import argparse, json, os, numpy as np, soundfile as sf, librosa
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import wilcoxon
from common import ROOT, SR, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gen", default="gen"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--eras", default="neutral,1960s,1980s,2010s"); a = ap.parse_args()
M = os.path.join(ROOT, "ess_models"); emb_m = TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(M, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
head = TensorflowPredict2D(graphFilename=os.path.join(M, "genre_discogs400-discogs-effnet-1.pb"), input="serving_default_model_Placeholder", output="PartitionedCall:0")
tags = json.load(open(os.path.join(M, "genre_discogs400-discogs-effnet-1.json")))["classes"]
def feat(y24):
    y = librosa.resample(np.asarray(y24, dtype=np.float32), orig_sr=SR, target_sr=16000); e = emb_m(y); t = head(e); return e.mean(0), t.mean(0)
rows = manifest("A"); lab = LABELS["A"]; split = np.array([r["split"] for r in rows]); y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows])
E, T = [], []
for i, r in enumerate(rows):
    e, t = feat(np.load(os.path.join(ROOT, "stems_codec", "A", r["sample_id"] + ".npz"))["accomp"]); E.append(e); T.append(t)
    if i % 200 == 0: print("train/val embed", i, len(rows), flush=True)
E, T = np.array(E), np.array(T); tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
best = max(((cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), E[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean(), C) for C in (0.001, 0.01, 0.1, 1.0)))
probe = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000)).fit(E[tr], y[tr]); acc = float((probe.predict(E[va]) == y[va]).mean()); print("probe cv", best, "val acc", acc, flush=True)
G = os.path.join(ROOT, a.gen); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; ERAS = a.eras.split(","); out = []; TT = {}
for s in ids:
    items = [("identity_codec", 0, np.load(os.path.join(ROOT, "stems_codec", "A", s + ".npz"))["accomp"])]
    for e in ERAS:
        for sd in a.seeds: items.append((e, sd, sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0]))
    for c, sd, w in items:
        e_, t_ = feat(w); out.append({"clip": s, "cond": c, "seed": sd, "probs": probe.predict_proba(e_[None])[0].tolist(), "tags": t_.tolist()})
    print(s, flush=True)
top = {}
for c in ERAS:
    m = np.mean([r["tags"] for r in out if r["cond"] == c], 0); n = np.mean([r["tags"] for r in out if r["cond"] == "neutral"], 0); top[c] = [(tags[i], round(float(m[i] - n[i]), 3)) for i in np.argsort(-(m - n))[:8]]
json.dump({"probe_cv": best, "probe_val_acc": acc, "rows": out, "top_tags_vs_neutral": top}, open(a.out, "w"))
