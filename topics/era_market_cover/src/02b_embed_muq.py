"""MuQ-large-msd-iter mean-pooled hidden states per layer, same conditions as 02_embed.py.  saves emb/<ds>_muq_<cond>.npy (N, L, 1024) fp16.
MuQ expects 24 kHz mono input (our native rate)."""
import argparse, os, numpy as np, torch, soundfile as sf, pyloudnorm as pyln
from common import ROOT, SR, manifest
from muq_util import load_muq
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conds", nargs="+", default=["mix_norm", "voc_norm", "acc_norm", "mix_raw", "acc_raw", "voc_raw", "recomb_norm"]); a = ap.parse_args()
m = load_muq(); meter = pyln.Meter(SR)
def norm(y, t=-23.0):
    if np.abs(y).max() < 1e-4: return y
    l = meter.integrated_loudness(y.astype(np.float64))
    return y if not np.isfinite(l) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, t), -1, 1).astype(np.float32)
rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows
def wave(r, c):
    if c.startswith("mix"): y = sf.read(r["path"], dtype="float32")[0]
    elif c.startswith(("accharm", "accperc")):
        y = np.load(os.path.join(ROOT, "stems_derived", "harm" if c.startswith("accharm") else "perc", a.ds, r["sample_id"] + ".npz"))["x"].astype(np.float32)
    elif c.startswith("acccodec"): y = np.load(os.path.join(ROOT, "stems_codec", a.ds, r["sample_id"] + ".npz"))["accomp"].astype(np.float32)
    else:
        z = np.load(os.path.join(ROOT, "stems_bsr" if c.startswith("bsr_") else "stems", a.ds, r["sample_id"] + ".npz")); v = z["vocals"].astype(np.float32); ac = z["accomp"].astype(np.float32)
        k = c[4:] if c.startswith("bsr_") else c; y = v if k.startswith("voc") else ac if k.startswith("acc") else v + ac
    return norm(y) if c.endswith("norm") else y
for c in a.conds:
    out = os.path.join(ROOT, "emb", f"{a.ds}_muq_{c}.npy")
    if os.path.exists(out) and not a.limit: continue
    E = []
    for i, r in enumerate(rows):
        x = torch.from_numpy(wave(r, c))[None].cuda()
        with torch.inference_mode(): o = m(x, output_hidden_states=True)
        E.append(torch.stack([h.float().mean(1) for h in o["hidden_states"]]).squeeze(1).cpu().numpy().astype(np.float16))
        if i % 300 == 0: print(a.ds, c, i, len(rows), E[-1].shape, flush=True)
    E = np.stack(E); assert np.isfinite(E.astype(np.float32)).all(), f"non-finite embeddings for {c}"
    if not a.limit: np.save(out, E)
    print("saved", c, E.shape, flush=True)
