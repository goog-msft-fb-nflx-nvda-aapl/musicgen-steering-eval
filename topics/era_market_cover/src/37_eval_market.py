"""Market-targeted cover evaluation (dataset B): accompaniment regenerated with market prompts (07_generate.py --ds B, GENDIR=gen_market), scored with market probes (MERT, MuQ) trained on EnCodec-resynthesized dataset-B TRAIN accompaniment
(layers from probe_codec_{mert,muq}_B.json, C=0.01 as in eraprobe). Delta_X = P(market X | prompt X) - P(market X | neutral prompt) per song, seeds averaged, Wilcoxon over songs.
usage (server venv): GENDIR=gen_market python 37_eval_market.py --out results/market_eval.json"""
import argparse, json, os, numpy as np, soundfile as sf
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, LABELS, manifest
from eraprobe import embed_waves, predict
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen_market")); picked = [l.split(",") for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; ids = [p[0] for p in picked]; true = {p[0]: p[1] for p in picked}
MK = LABELS["B"]; PR = ["neutral"] + MK
rows = manifest("B"); y = np.array([MK.index(r["label"]) if r["label"] else -1 for r in rows]); tr = np.where(np.array([r["split"] for r in rows]) == "train")[0]
layer = {k: json.load(open(os.path.join(ROOT, "results", f"probe_codec_{k}_B.json")))["conds"]["acccodec_norm"]["layer"] for k in ("mert", "muq")}; probes = {}
for k, pre in (("mert", ""), ("muq", "muq_")):
    E = np.load(os.path.join(ROOT, "emb", f"B_{pre}acccodec_norm.npy")).astype(np.float32)[:, layer[k]]; probes[k] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(E[tr], y[tr])
clips = {}
for s in ids:
    clips[f"{s}|identity_codec|0"] = np.load(os.path.join(ROOT, "stems_codec", "B", s + ".npz"))["accomp"].astype(np.float32)
    for e in PR:
        for sd in a.seeds: clips[f"{s}|{e}|{sd}"] = sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0]
emb = embed_waves(clips, layer); P = predict(probes, emb); R = []
for n in clips:
    s, c, sd = n.split("|"); R.append({"clip": s, "true": true[s], "cond": c, "seed": int(sd), "mert_probs": P["mert"][n].tolist(), "muq_probs": P["muq"][n].tolist()})
json.dump({"layers": layer, "rows": R}, open(a.out, "w")); songs = sorted(set(ids))
for enc in ("mert", "muq"):
    print(f"\n== {enc}: Delta P(market X | prompt X) - P(X | neutral), mean over {len(songs)} songs [songs>0], wilcoxon p; and P(X|prompt X)")
    for t, m in enumerate(MK):
        Pm = lambda c: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == s and r["cond"] == c]) for s in songs]); d = Pm(m) - Pm("neutral")
        print(f"  {m:8s} {d.mean():+.3f} [{int((d > 0).sum())}/{len(songs)}] p={wilcoxon(d).pvalue if np.any(d != 0) else 1:.4f}  P={Pm(m).mean():.3f} (neutral {Pm('neutral').mean():.3f}, chance 0.167)")
    print("  identity_codec probe acc vs true market:", np.mean([int(np.argmax(r[f"{enc}_probs"]) == MK.index(r["true"])) for r in R if r["cond"] == "identity_codec"]))
