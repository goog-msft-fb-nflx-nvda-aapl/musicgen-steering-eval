"""COCOLA (Coherence-Oriented Contrastive Learning of Musical Audio Representations, COCOLA_HP_v1 checkpoint) between the ORIGINAL vocal stem and an accompaniment, as a vocal-accompaniment fit score.
5 s windows at 16 kHz (6 non-overlapping windows per 30 s clip, mean), modes BOTH / HARMONIC / PERCUSSIVE. Pairs per song: original vocal x {original accompaniment, generated accompaniment per prompt/seed (gen/), codec-resynthesized original,
and mismatched floor: the same generated accompaniment against ANOTHER song's vocal}. run from ~/era_market/ext/cocola:  python ../../src/24_cocola.py --out ../../results/cocola.json"""
import argparse, json, os, sys, numpy as np, soundfile as sf, torch, torchaudio.functional as AF
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from contrastive_model import constants
from contrastive_model.contrastive_model import CoCola
from feature_extraction.feature_extraction import CoColaFeatureExtractor
from common import ROOT, SR
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--gen", default="mcl_gen"); a = ap.parse_args()
model = CoCola.load_from_checkpoint("ckpt/cocola_hp_v1.ckpt", map_location="cuda").eval().cuda(); fe = CoColaFeatureExtractor()
G = os.path.join(ROOT, a.gen); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
def win(y):   # 24 kHz mono -> (6,1,80000) at 16 kHz
    x = AF.resample(torch.from_numpy(np.asarray(y, dtype=np.float32)), SR, 16000); n = len(x) // 80000; return x[: n * 80000].reshape(n, 1, 80000)
def score(fv, fy, mode):   # fv, fy: precomputed COCOLA features
    model.set_embedding_mode(mode)
    with torch.no_grad(): return float(model.score(fv.clone(), fy.clone()).mean())   # clone: set_embedding_mode masks channels in place
MODES = {"both": constants.EmbeddingMode.BOTH, "harm": constants.EmbeddingMode.HARMONIC, "perc": constants.EmbeddingMode.PERCUSSIVE}
voc, acc = {}, {}; FV = {}
for s in ids:
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc[s] = win(z["vocals"]); acc[s] = win(z["accomp"])
for s in ids: FV[s] = fe(voc[s]).cuda()
codec = {s: win(np.load(os.path.join(ROOT, "stems_codec", "A", s + ".npz"))["accomp"]) for s in ids}
rows = []
for i, s in enumerate(ids):
    other = ids[(i + 1) % len(ids)]
    items = [("original", 0, acc[s]), ("original_codec", 0, codec[s])]
    for e in ("neutral", "1960s", "1980s", "2010s"):
        for sd in a.seeds: items.append((e, sd, win(sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0])))
    for cond, sd, y in items:
        r = {"clip": s, "cond": cond, "seed": sd}; fy = fe(y[: len(voc[s])]).cuda()   # all clips have 6 windows
        for m, mode in MODES.items(): r[m] = score(FV[s], fy, mode); r[m + "_mismatch"] = score(FV[other], fy, mode)
        rows.append(r)
    print(s, flush=True)
json.dump(rows, open(a.out, "w"))
