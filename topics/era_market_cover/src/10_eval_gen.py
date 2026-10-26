"""Evaluate the MusicGen-Melody era regenerations (gen/ from 07_generate.py).
per generated clip: (a) melody/harmony preservation = chroma (CQT, 12 bins) cosine between the original MIX and the generated REMIX, frame-aligned and DTW-aligned, plus a control (original mix vs the remix of a different song);
(b) era presence = decade probe trained on dataset-A TRAIN accompaniment stems (acc_norm, layer from probe json) applied to the generated accompaniment (loudness-normalized);
the probe is run with two encoders (MERT-v2-30s, MuQ); report expected decade index and probability mass on the target decade.
baselines: identity (original accompaniment stem), DSP "old" (band-limit 60 Hz-7 kHz, tape-style soft clipping, hiss) and DSP "new" (soft limiter, high shelf) applied to the original accompaniment.
usage: python 10_eval_gen.py --out results/gen_eval.json"""
import argparse, json, os, numpy as np, torch, soundfile as sf, librosa, pyloudnorm as pyln
from scipy.signal import butter, sosfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transformers import AutoFeatureExtractor, AutoModel
from common import ROOT, SR, manifest, LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--train_cond", default="acc_norm"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--probe_mert", default="probe_A.json"); ap.add_argument("--probe_muq", default="probe_muq_A.json"); a = ap.parse_args()
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen")); picked = [l.split(",") for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
ids = [p[0] for p in picked]; true = {p[0]: p[1] for p in picked}; ERAS = ["neutral", "1960s", "1980s", "2010s"]; SEEDS = a.seeds
rows = manifest("A"); rid = {r["sample_id"]: i for i, r in enumerate(rows)}
meter = pyln.Meter(SR)
def norm(y):
    if np.abs(y).max() < 1e-4: return y.astype(np.float32)
    l = meter.integrated_loudness(y.astype(np.float64)); return y.astype(np.float32) if not np.isfinite(l) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
def dsp_old(y):
    y = sosfilt(butter(4, [60, 7000], btype="band", fs=SR, output="sos"), y); y = np.tanh(1.8 * y) / 1.8
    return (y + np.random.default_rng(0).normal(0, 10 ** (-50 / 20), len(y))).astype(np.float32)
def dsp_new(y):
    y = y / (np.abs(y).max() + 1e-9); y = np.tanh(3.0 * y) / np.tanh(3.0)
    return (y + 0.6 * sosfilt(butter(2, 3000, btype="high", fs=SR, output="sos"), y)).astype(np.float32)
clips = {}   # name -> waveform of accompaniment-like signal
meta = []
for s in ids:
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); acc = z["accomp"].astype(np.float32)
    clips[f"{s}|identity|0"] = acc; clips[f"{s}|identity_codec|0"] = np.load(os.path.join(ROOT, "stems_codec", "A", s + ".npz"))["accomp"].astype(np.float32); clips[f"{s}|dsp_old|0"] = dsp_old(acc); clips[f"{s}|dsp_new|0"] = dsp_new(acc)
    for e in ERAS:
        for sd in SEEDS: clips[f"{s}|{e}|{sd}"] = sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0]
names = list(clips); print(len(names), "clips", flush=True)
pj = {"mert": json.load(open(os.path.join(ROOT, "results", a.probe_mert))), "muq": json.load(open(os.path.join(ROOT, "results", a.probe_muq)))}
layer = {k: pj[k]["conds"][a.train_cond]["layer"] for k in pj}
split = np.array([r["split"] for r in rows]); y = np.array([LABELS["A"].index(r["label"]) if r["label"] else -1 for r in rows]); tr = np.where(split == "train")[0]
probes = {}
for k, pre in (("mert", ""), ("muq", "muq_")):
    E = np.load(os.path.join(ROOT, "emb", f"A_{pre}{a.train_cond}.npy")).astype(np.float32)[:, layer[k]]
    probes[k] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(E[tr], y[tr])
emb = {"mert": [], "muq": []}
proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True); mert = AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True).eval().cuda()
W = {n: norm(clips[n]) for n in names}
for n in names:   # MERT first, before the muq package is imported (importing it changed MERT outputs to NaN when both were loaded)
    inp = proc(W[n], sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float(); h = o.hidden_states[layer["mert"]]; emb["mert"].append(((h * m).sum(1) / m.sum(1)).squeeze(0).cpu().numpy())
del mert; torch.cuda.empty_cache()
from muq_util import load_muq
muq = load_muq()
for n in names:
    with torch.inference_mode(): oq = muq(torch.from_numpy(W[n])[None].cuda(), output_hidden_states=True)
    emb["muq"].append(oq["hidden_states"][layer["muq"]].float().mean(1).squeeze(0).cpu().numpy())
P = {}
for k in emb:
    X = np.array(emb[k]); ok = np.isfinite(X).all(1); print(k, "non-finite embeddings:", [names[i] for i in np.where(~ok)[0]], flush=True)
    pr = np.full((len(X), 6), np.nan); 
    if ok.any(): pr[ok] = probes[k].predict_proba(X[ok])
    P[k] = pr   # (n, 6), classes ordered 0..5
def chroma(x): return librosa.feature.chroma_cqt(y=x, sr=SR, hop_length=1024)
def cosf(a_, b_):
    n = min(a_.shape[1], b_.shape[1]); a_, b_ = a_[:, :n], b_[:, :n]; return float(((a_ * b_).sum(0) / (np.linalg.norm(a_, axis=0) * np.linalg.norm(b_, axis=0) + 1e-9)).mean())
def dtw_cos(a_, b_):
    D, wp = librosa.sequence.dtw(X=a_, Y=b_, metric="cosine"); return float(1 - D[-1, -1] / len(wp))
mixC = {s: chroma(sf.read(rows[rid[s]]["path"], dtype="float32")[0]) for s in ids}
out = []
for i, n in enumerate(names):
    s, e, sd = n.split("|"); r = {"clip": s, "true": true[s], "cond": e, "seed": int(sd)}
    for k in P: r[f"{k}_exp_decade"] = float((P[k][i] * np.arange(6)).sum()); r[f"{k}_probs"] = np.round(P[k][i], 4).tolist()
    if e in ERAS:
        rm = sf.read(os.path.join(G, f"{s}_{e}_s{sd}_remix.wav"), dtype="float32")[0]; c = chroma(rm)
        other = [t for t in ids if t != s][(ids.index(s) * 3 + int(sd)) % (len(ids) - 1)]
        r["chroma_cos"] = cosf(mixC[s], c); r["chroma_dtw"] = dtw_cos(mixC[s], c); r["chroma_cos_other_song"] = cosf(mixC[other], c); r["chroma_dtw_other_song"] = dtw_cos(mixC[other], c)
    out.append(r)
np.savez(a.out.replace(".json", "_emb.npz"), names=np.array(names), mert=np.array(emb["mert"]), muq=np.array(emb["muq"]))
json.dump({"layers": layer, "rows": out}, open(a.out, "w")); print("saved", a.out, len(out))
