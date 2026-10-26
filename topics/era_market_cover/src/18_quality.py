"""Audiobox Aesthetics (facebook/audiobox-aesthetics: PQ production quality, PC production complexity, CE content enjoyment, CU content usefulness; reference-free, per clip) for generated accompaniment and baselines.
conditions: generated (neutral/1960s/1980s/2010s per seed), identity (original accompaniment stem), identity_codec, dsp_old, dsp_new, white/pink noise.  saves results/quality.json
usage: python 18_quality.py --seeds 0 1 2 --out results/quality.json"""
import argparse, json, os, numpy as np, soundfile as sf, tempfile
from scipy.signal import butter, sosfilt
from common import ROOT, SR
from audiobox_aesthetics.infer import initialize_predictor
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--out", required=True); a = ap.parse_args()
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen")); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
ERAS = ["neutral", "1960s", "1980s", "2010s"]; rng = np.random.default_rng(0); tmp = tempfile.mkdtemp(dir=ROOT)
def dsp_old(y):
    y = sosfilt(butter(4, [60, 7000], btype="band", fs=SR, output="sos"), y); y = np.tanh(1.8 * y) / 1.8; return (y + np.random.default_rng(0).normal(0, 10 ** (-50 / 20), len(y))).astype(np.float32)
def dsp_new(y):
    y = y / (np.abs(y).max() + 1e-9); y = np.tanh(3.0 * y) / np.tanh(3.0); return (y + 0.6 * sosfilt(butter(2, 3000, btype="high", fs=SR, output="sos"), y)).astype(np.float32)
items = []   # (name, path)
def add(name, y): p = os.path.join(tmp, name.replace("|", "_") + ".wav"); sf.write(p, np.clip(y, -1, 1), SR); items.append((name, p))
for s in ids:
    if not all(os.path.exists(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav")) for e in ERAS for sd in a.seeds): continue
    acc = np.load(os.path.join(ROOT, "stems", "A", s + ".npz"))["accomp"].astype(np.float32)
    add(f"{s}|identity|0", acc); add(f"{s}|identity_codec|0", np.load(os.path.join(ROOT, "stems_codec", "A", s + ".npz"))["accomp"].astype(np.float32)); add(f"{s}|dsp_old|0", dsp_old(acc)); add(f"{s}|dsp_new|0", dsp_new(acc))
    for e in ERAS:
        for sd in a.seeds: items.append((f"{s}|{e}|{sd}", os.path.join(G, f"{s}_{e}_s{sd}_acc.wav")))
for i in range(12): add(f"noise|white|{i}", rng.normal(0, .1, 30 * SR).astype(np.float32))
p = initialize_predictor(); res = {}
for i in range(0, len(items), 16):
    b = items[i:i + 16]; out = p.forward([{"path": f} for _, f in b])
    for (n, _), o in zip(b, out): res[n] = {k: float(v) for k, v in o.items()}
json.dump(res, open(a.out, "w")); print("saved", len(res))
