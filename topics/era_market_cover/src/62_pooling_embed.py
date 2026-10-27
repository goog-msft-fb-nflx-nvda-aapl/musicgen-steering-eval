"""Pooling ablation, step 1 (GPU): re-embed every clip (HTDemucs stems, loudness-normalized -23 LUFS, same conditions as 02_embed.py) with MERT-v2-30s and save time-pooling statistics besides the mean:
emb_pool/<ds>_<cond>_{std,max}.npy (N, 25, 1024) float16 = per-layer standard deviation / maximum over time frames; emb_pool/<ds>_<cond>_seg3.npy (N, 3, 25, 1024) = mean over each third of the clip (10 s).
conds: mix_norm, voc_norm, acc_norm. usage: python 62_pooling_embed.py A|B"""
import sys, os, numpy as np, torch, soundfile as sf, pyloudnorm as pyln
from common import ROOT, SR, manifest
from mert_util import load_mert
ds = sys.argv[1]; rows = manifest(ds); proc, model = load_mert(); meter = pyln.Meter(SR); os.makedirs(os.path.join(ROOT, "emb_pool"), exist_ok=True)
def norm(y, t=-23.0):
    if np.abs(y).max() < 1e-4: return y
    l = meter.integrated_loudness(y.astype(np.float64)); return y if not np.isfinite(l) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, t), -1, 1).astype(np.float32)
def wave(r, cond):
    if cond == "mix_norm": return norm(sf.read(r["path"], dtype="float32")[0])
    z = np.load(os.path.join(ROOT, "stems", ds, r["sample_id"] + ".npz")); return norm((z["vocals"] if cond == "voc_norm" else z["accomp"]).astype(np.float32))
for cond in ("mix_norm", "voc_norm", "acc_norm"):
    out = [os.path.join(ROOT, "emb_pool", f"{ds}_{cond}_{k}.npy") for k in ("std", "max", "seg3")]
    if all(os.path.exists(o) for o in out): continue
    S, M, G = [], [], []
    for i, r in enumerate(rows):
        inp = proc(wave(r, cond), sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
        with torch.inference_mode(): o = model(**inp, output_hidden_states=True)
        H = torch.stack(list(o.hidden_states)).squeeze(1).float()          # (25, T, 1024)
        T = H.shape[1]; S.append(H.std(1).cpu().numpy().astype(np.float16)); M.append(H.max(1).values.cpu().numpy().astype(np.float16))
        G.append(np.stack([H[:, a * T // 3:(a + 1) * T // 3].mean(1).cpu().numpy() for a in range(3)]).astype(np.float16))
        if i % 300 == 0: print(ds, cond, i, len(rows), T, flush=True)
    for o, v in zip(out, (S, M, G)): np.save(o, np.stack(v))
