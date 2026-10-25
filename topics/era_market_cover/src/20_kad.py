"""KAD-style distribution metric (own implementation): unbiased MMD^2 with an RBF kernel (median heuristic on the pooled sample) between generated accompaniment embeddings and codec-matched REAL accompaniment references of each decade (dataset-A TRAIN stems through EnCodec, MERT layer / MuQ layer as in the codec probes).
 own-era margin for a prompt X (target decade t) = mean_{d != t} MMD2(gen_X, ref_d) - MMD2(gen_X, ref_t)   (> 0 means closer to the target decade than to the others)
 reported as margin(prompt X) - margin(neutral prompt), bootstrap over songs (generated side resampled by song, references fixed).
usage: python 20_kad.py results/gen_eval_codec_24_emb.npz --out results/kad_24.json"""
import argparse, json, os, numpy as np
from common import ROOT, manifest, LABELS
ap = argparse.ArgumentParser(); ap.add_argument("emb"); ap.add_argument("--out", required=True); ap.add_argument("--layers", default=os.path.join(ROOT, "results", "gen_eval_codec_24.json")); a = ap.parse_args()
Z = np.load(a.emb); names = list(Z["names"]); layer = json.load(open(a.layers))["layers"]
rows = manifest("A"); y = np.array([LABELS["A"].index(r["label"]) if r["label"] else -1 for r in rows]); tr = np.array([r["split"] == "train" for r in rows])
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; rng = np.random.default_rng(0)
def mmd2(X, Y, sig):
    def k(A, B): d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1); return np.exp(-d / (2 * sig ** 2))
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y); n, m = len(X), len(Y)
    return (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean()
res = {}
for enc, pre in (("mert", ""), ("muq", "muq_")):
    ref = np.load(os.path.join(ROOT, "emb", f"A_{pre}acccodec_norm.npy")).astype(np.float32)[:, layer[enc]]; R = {d: ref[tr & (y == d)] for d in range(6)}
    mu, sd = ref[tr].mean(0), ref[tr].std(0) + 1e-6; nz = lambda v: (v - mu) / sd
    G = {n: nz(v) for n, v in zip(names, Z[enc])}; R = {d: nz(v) for d, v in R.items()}
    pool = np.concatenate([R[d] for d in R] + [np.array(list(G.values()))]); idx = rng.choice(len(pool), 600, replace=False); pp = pool[idx]
    sig = float(np.sqrt(np.median(((pp[:, None] - pp[None]) ** 2).sum(-1)[np.triu_indices(600, 1)]) / 2)); songs = sorted({n.split("|")[0] for n in names})
    def margin(cond, ss, t):
        X = np.array([G[n] for n in names if n.split("|")[0] in ss and n.split("|")[1] == cond]); others = [mmd2(X, R[d], sig) for d in range(6) if d != t]
        return float(np.mean(others) - mmd2(X, R[t], sig))
    res[enc] = {"sigma": sig}
    for e, t in TD.items():
        full = margin(e, set(songs), t) - margin("neutral", set(songs), t)
        bs = [margin(e, set(rng.choice(songs, len(songs))), t) - margin("neutral", set(rng.choice(songs, len(songs))), t) for _ in range(200)]
        res[enc][e] = {"margin": margin(e, set(songs), t), "neutral_margin": margin("neutral", set(songs), t), "delta": full, "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        print(f"{enc} {e}: own-era margin {res[enc][e]['margin']:+.4f}, neutral {res[enc][e]['neutral_margin']:+.4f}, delta {full:+.4f} [{res[enc][e]['ci95'][0]:+.4f}, {res[enc][e]['ci95'][1]:+.4f}]")
    for c in ("identity_codec", "dsp_old", "dsp_new"):
        print(f"  {enc} baseline {c}: margins " + ", ".join(f"{e} {margin(c, set(songs), t):+.4f}" for e, t in TD.items()))
json.dump(res, open(a.out, "w"))
