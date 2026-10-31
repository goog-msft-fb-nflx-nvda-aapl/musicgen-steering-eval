"""Probe augmentation with a second EnCodec pass (follow-up to 73_probe_augment.py).
Dataset-A accompaniment stems (all 1290 clips) are passed through the MusicGen EnCodec (2.2 kbps) twice (stems_codec -> codec again), loudness-normalized and embedded (MERT L18 / MuQ L9, the codec-probe layers). Probes: codec (main) vs codec+double (both sets of TRAIN clips, 2x) and double only; validation accuracy on codec-resynthesized validation stems; Delta of the stored generated-set embeddings (73_probe_augment.py statistic) for each.
usage: python 77_probe_doublecodec.py set1 set2 ... -> results/probe_doublecodec.{json,txt}, emb/A_{,muq_}acc2codec_norm_L.npy"""
import sys, os, json, numpy as np, torch, torchaudio.functional as AF
from transformers import EncodecModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, SR, LABELS, manifest
import eraprobe as EP
rows = manifest("A"); lab = LABELS["A"]; y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
PJ = {"mert": json.load(open(f"{ROOT}/results/probe_codec_mert_A.json")), "muq": json.load(open(f"{ROOT}/results/probe_codec_muq_A.json"))}; L = {k: PJ[k]["conds"]["acccodec_norm"]["layer"] for k in PJ}
fp = {k: f"{ROOT}/emb/A_{'muq_' if k == 'muq' else ''}acc2codec_norm_L.npy" for k in PJ}
if not all(os.path.exists(f) for f in fp.values()):
    enc = EncodecModel.from_pretrained("facebook/encodec_32khz").cuda().eval(); W = {}
    for i, r in enumerate(rows):
        a = np.load(os.path.join(ROOT, "stems_codec", "A", r["sample_id"] + ".npz"))["accomp"].astype(np.float32); x = AF.resample(torch.from_numpy(a), SR, 32000)[None, None].cuda()
        with torch.no_grad(): e = enc.encode(x, bandwidth=2.2); d = enc.decode(e.audio_codes, e.audio_scales)[0]
        z = AF.resample(d[0, 0].float().cpu(), 32000, SR).numpy()[:len(a)]; W[str(i)] = np.pad(z, (0, len(a) - len(z))).astype(np.float32)
    del enc; torch.cuda.empty_cache(); em = EP.embed_waves(W, L)
    for k in PJ: np.save(fp[k], np.array([em[k][str(i)] for i in range(len(rows))]))
D = {k: np.load(fp[k]).astype(np.float32) for k in PJ}; C = {k: np.load(f"{ROOT}/emb/A_{'muq_' if k == 'muq' else ''}acccodec_norm.npy").astype(np.float32)[:, L[k]] for k in PJ}
def fit(k, kind):
    if kind == "codec": X, yy = C[k][tr], y[tr]
    elif kind == "double": X, yy = D[k][tr], y[tr]
    else: X, yy = np.concatenate([C[k][tr], D[k][tr]]), np.concatenate([y[tr], y[tr]])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X, yy)
KINDS = ("codec", "double", "codec+double"); M = {(k, kd): fit(k, kd) for k in PJ for kd in KINDS}; rng = np.random.default_rng(0); TD = {"1960s": 0, "1980s": 2, "2010s": 5}; out = {"val_acc": {}, "sets": {}}; txt = []
for k in PJ:
    for kd in KINDS: out["val_acc"][f"{k}_{kd}"] = {"codec_val": float((M[(k, kd)].predict(C[k][va]) == y[va]).mean()), "double_val": float((M[(k, kd)].predict(D[k][va]) == y[va]).mean())}
txt.append("validation accuracy (codec-resynthesized / double-codec validation stems):\n" + "\n".join(f"  {key}: {v['codec_val']:.3f} / {v['double_val']:.3f}" for key, v in out["val_acc"].items()))
for s in sys.argv[1:]:
    R = json.load(open(f"{ROOT}/results/{s}_eval.json"))["rows"]; Z = np.load(f"{ROOT}/results/{s}_eval_emb.npz"); songs = sorted({r["clip"] for r in R}); res = {}
    for k in PJ:
        X = Z[k]; ok = np.isfinite(X).all(1)
        for kd in KINDS:
            pr = np.full((len(X), 6), np.nan); pr[ok] = M[(k, kd)].predict_proba(X[ok]); cells = {}
            for e, t in TD.items():
                P = lambda c: np.array([np.mean([pr[i, t] for i, r in enumerate(R) if r["clip"] == x and r["cond"] == c]) for x in songs]); d = P(e) - P("neutral"); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                cells[e] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            res[f"{k}_{kd}"] = cells
    out["sets"][s] = res; txt.append(s + "\n" + "\n".join(f"  {key:18s} " + " | ".join(f"{e} {c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]" for e, c in cells.items()) for key, cells in res.items()))
json.dump(out, open(f"{ROOT}/results/probe_doublecodec.json", "w"), indent=1); open(f"{ROOT}/results/probe_doublecodec.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
