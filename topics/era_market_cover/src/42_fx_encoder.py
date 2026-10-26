"""Production-only evaluator: Fx-Encoder++ (Sony, SimCLR audio-effects representation; 2048-d pre-projection embedding, mono duplicated to stereo, 44.1 kHz, three 10 s chunks averaged). Run in venv_fx (GPU).
(1) embeds the EnCodec-resynthesized real accompaniment of all dataset-A clips; decade probe (logreg, C by 5-fold CV on train, validation scored once) = how much decade information production/effects features carry;
(2) Delta P(target decade | prompt) - P(target | neutral) for generators (dirs given by --gens name=dir:seeds) and the production-chain control (gen_fx). usage: python 42_fx_encoder.py --out results/fx_encoder.json"""
import argparse, json, os, numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from fxencoder_plusplus import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import wilcoxon
from common import ROOT, SR, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
model = load_model("default", device="cuda"); model.eval()
def emb(y24):
    x = AF.resample(torch.from_numpy(np.asarray(y24, dtype=np.float32)), SR, 44100); n = max(len(x) // 441000, 1); v = []
    for k in range(n):
        c = x[k * 441000:(k + 1) * 441000]; c = c[None, None].repeat(1, 2, 1).cuda()
        with torch.no_grad(): v.append(model.fx_encoder(c)["embedding"][0].float().cpu().numpy())
    return np.mean(v, 0)
rows = manifest("A"); lab = LABELS["A"]; sp = np.array([r["split"] for r in rows]); y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); tr, va = np.where(sp == "train")[0], np.where(sp == "validation")[0]
E = np.array([emb(np.load(os.path.join(ROOT, "stems_codec", "A", r["sample_id"] + ".npz"))["accomp"]) for r in rows]); print("embedded", E.shape, flush=True)
Eo = np.array([emb(np.load(os.path.join(ROOT, "stems", "A", r["sample_id"] + ".npz"))["accomp"]) for r in rows if r["split"] == "validation"])
best = max((cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), E[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean(), C) for C in (0.001, 0.01, 0.1, 1.0))
probe = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=3000)).fit(E[tr], y[tr]); acc = float((probe.predict(E[va]) == y[va]).mean()); acc_orig = float((probe.predict(Eo) == y[va]).mean()); print("fx probe cv", best, "val acc (codec acc)", acc, "(original acc)", acc_orig, flush=True)
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; res = {"probe_cv": best, "val_acc_codec": acc, "val_acc_original": acc_orig, "gens": {}}
for name, d, seeds in (("musicgen", "gen", [0, 1, 2]), ("musecontrol", "mcl_gen", [0, 1, 2]), ("textonly", "gen_textonly", [0, 1, 2]), ("fxchain", "gen_fx", [0])):
    G = os.path.join(ROOT, d); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; P = {}
    for s in ids:
        for e in ("neutral", "1960s", "1980s", "2010s"):
            P[(s, e)] = np.mean([probe.predict_proba(emb(sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0])[None])[0] for sd in seeds], 0)
    out = {}
    for e, t in TD.items():
        dl = np.array([P[(s, e)][t] - P[(s, "neutral")][t] for s in ids]); out[e] = {"delta": float(dl.mean()), "pos": int((dl > 0).sum()), "p": float(wilcoxon(dl).pvalue) if np.any(dl != 0) else 1.0}
    res["gens"][name] = out; print(name, {e: (round(v["delta"], 3), v["pos"], round(v["p"], 4)) for e, v in out.items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
