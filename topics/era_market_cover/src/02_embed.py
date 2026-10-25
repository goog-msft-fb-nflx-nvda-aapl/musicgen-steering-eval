"""MERT-v2-30s (m-a-p/MERT-v2-30s) mean-pooled hidden states per layer for each condition.
conditions: mix_raw, mix_norm, voc_raw, voc_norm, acc_raw, acc_norm, recomb_norm  (norm = loudness to -23 LUFS, pyloudnorm)
saves emb/<ds>_<cond>.npy  (N, L, 1024) float16 in manifest order.  usage: python 02_embed.py --ds A --conds mix_raw mix_norm ..."""
import argparse, os, numpy as np, torch, soundfile as sf, pyloudnorm as pyln
from transformers import AutoFeatureExtractor, AutoModel
from common import ROOT, SR, manifest

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conds", nargs="+", default=["mix_raw", "mix_norm", "voc_raw", "voc_norm", "acc_raw", "acc_norm", "recomb_norm"])
a = ap.parse_args()
mid = "m-a-p/MERT-v2-30s"
proc = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=True)
model = AutoModel.from_pretrained(mid, trust_remote_code=True).eval().cuda()
meter = pyln.Meter(SR)
def norm(y, target=-23.0):
    if np.abs(y).max() < 1e-4: return y
    l = meter.integrated_loudness(y.astype(np.float64))
    if not np.isfinite(l): return y
    return np.clip(pyln.normalize.loudness(y.astype(np.float64), l, target), -1, 1).astype(np.float32)
rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows
os.makedirs(os.path.join(ROOT, "emb"), exist_ok=True)

def wave(r, cond):
    if cond.startswith("mix"): y = sf.read(r["path"], dtype="float32")[0]
    elif cond.startswith("acccodec"): y = np.load(os.path.join(ROOT, "stems_codec", a.ds, r["sample_id"] + ".npz"))["accomp"].astype(np.float32)
    else:
        z = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz")); v = z["vocals"].astype(np.float32); c = z["accomp"].astype(np.float32)
        y = v if cond.startswith("voc") else c if cond.startswith("acc") else v + c
    return norm(y) if cond.endswith("norm") else y

for cond in a.conds:
    out = os.path.join(ROOT, "emb", f"{a.ds}_{cond}.npy")
    if os.path.exists(out) and not a.limit: continue
    E = []
    for i, r in enumerate(rows):
        y = wave(r, cond)
        inp = proc(y, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
        with torch.inference_mode(): o = model(**inp, output_hidden_states=True)
        m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
        hs = torch.stack([((h * m).sum(1) / m.sum(1).clamp_min(1)) if m is not None else h.mean(1) for h in o.hidden_states]).squeeze(1)
        E.append(hs.cpu().numpy().astype(np.float16))
        if i % 200 == 0: print(a.ds, cond, i, len(rows), hs.shape, flush=True)
    E = np.stack(E)
    if not a.limit: np.save(out, E)
    print("saved", cond, E.shape, flush=True)
