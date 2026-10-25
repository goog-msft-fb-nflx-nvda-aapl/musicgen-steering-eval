"""Validation controls for the era metric (SURVEY_metrics.md section 4), codec-matched probes.
 - sensitivity curve: crossfade original (codec-resynthesized) accompaniment -> generated accompaniment, alpha in 0, .25, .5, .75, 1, for each song and era prompt (seed 0)
 - negative controls: pink noise, white noise (-23 LUFS), and the generated audio of a DIFFERENT song (label swap)
saves results/controls.json  usage: python 16_controls.py --out results/controls.json"""
import argparse, json, os, numpy as np, soundfile as sf
from common import ROOT, SR
from eraprobe import load_probes, embed_waves, predict
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0]); a = ap.parse_args()
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
ids = [s for s in ids if all(os.path.exists(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav")) for e in ("1960s", "1980s", "2010s") for sd in a.seeds)]; print(len(ids), "songs with all generations", flush=True)
probes, layer = load_probes(); waves = {}; rng = np.random.default_rng(0); ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
def rms(x): return float(np.sqrt((x ** 2).mean()) + 1e-9)
for s in ids:
    orig = np.load(os.path.join(ROOT, "stems_codec", "A", s + ".npz"))["accomp"].astype(np.float32)
    for e in ("1960s", "1980s", "2010s"):
        for sd in a.seeds:
            g = sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0]; g = g * (rms(orig) / rms(g))
            for al in ALPHAS: waves[f"sens|{s}|{e}|{sd}|{al}"] = (1 - al) * orig + al * g
for i in range(12):
    white = rng.normal(0, 1, 30 * SR).astype(np.float32); f = np.fft.rfft(rng.normal(0, 1, 30 * SR)); f /= np.sqrt(np.maximum(np.arange(len(f)), 1)); pink = np.fft.irfft(f).astype(np.float32)
    waves[f"noise|white|{i}"] = white; waves[f"noise|pink|{i}"] = pink
emb = embed_waves(waves, layer); P = predict(probes, emb)
json.dump({"layers": layer, "alphas": ALPHAS, "probs": {k: {n: v.round(5).tolist() for n, v in P[k].items()} for k in P}}, open(a.out, "w")); print("saved", len(waves))
