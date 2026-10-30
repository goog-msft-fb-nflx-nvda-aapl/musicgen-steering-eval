"""Which part of the crude DSP "old"/"new" baselines moves the decade probes (10_eval_gen.py baselines, split into components).
Applied to the 24 validation songs' original accompaniment stems (dataset A, stems/A): old = band-limit 60 Hz-7 kHz -> tanh soft clip (drive 1.8) -> white hiss at -50 dB; new = peak-normalize -> tanh limiter (drive 3) -> + 0.6 x high-passed (3 kHz) copy.
Components alone: bandlimit, clip, hiss, limiter, shelf.  Scoring: codec-matched probes (eraprobe.py: MERT L18, MuQ L9, trained on EnCodec-resynthesized TRAIN stems), loudness-normalized input; statistic = per-song change of expected decade index (0=1960s .. 5=2010s) versus the unprocessed stem, Wilcoxon over songs, and the change of P(target decade) for 1960s, 1980s, 2010s.
usage: python 74_dsp_components.py -> results/dsp_components.json, results/dsp_components.txt"""
import os, json, numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon
from common import ROOT, SR
import eraprobe as EP
ids = [l.split(",")[0] for l in open(os.path.join(ROOT, "gen", "picked.txt")).read().split("\n") if l.strip()]
def bandlimit(y): return sosfilt(butter(4, [60, 7000], btype="band", fs=SR, output="sos"), y)
def clip(y): return np.tanh(1.8 * y) / 1.8
def hiss(y): return y + np.random.default_rng(0).normal(0, 10 ** (-50 / 20), len(y))
def limiter(y): y = y / (np.abs(y).max() + 1e-9); return np.tanh(3.0 * y) / np.tanh(3.0)
def shelf(y): return y + 0.6 * sosfilt(butter(2, 3000, btype="high", fs=SR, output="sos"), y)
V = {"identity": lambda y: y, "bandlimit": bandlimit, "clip": clip, "hiss": hiss, "limiter": limiter, "shelf": shelf, "old": lambda y: hiss(clip(bandlimit(y))), "new": lambda y: shelf(limiter(y))}
waves = {}
for s in ids:
    acc = np.load(os.path.join(ROOT, "stems", "A", s + ".npz"))["accomp"].astype(np.float32)
    for v, f in V.items(): waves[f"{s}|{v}"] = np.asarray(f(acc), dtype=np.float32)
probes, layer = EP.load_probes(); emb = EP.embed_waves(waves, layer); pr = EP.predict(probes, emb); TD = {"1960s": 0, "1980s": 2, "2010s": 5}; out = {"songs": ids, "layers": layer, "variants": {}}; txt = []
for k in ("mert", "muq"):
    P = {v: np.array([pr[k][f"{s}|{v}"] for s in ids]) for v in V}; ed = {v: (P[v] * np.arange(6)).sum(1) for v in V}
    for v in V:
        if v == "identity": continue
        d = ed[v] - ed["identity"]; r = {"d_expected_decade": float(d.mean()), "pos": int((d > 0).sum()), "p": float(wilcoxon(d).pvalue)}
        for e, t in TD.items(): r[f"d_P_{e}"] = float((P[v][:, t] - P["identity"][:, t]).mean())
        out["variants"][f"{k}_{v}"] = r; txt.append(f"{k:5s} {v:10s} d(expected decade) {r['d_expected_decade']:+.3f} [{r['pos']}/{len(ids)}] p={r['p']:.4f} | dP 1960s {r['d_P_1960s']:+.3f} 1980s {r['d_P_1980s']:+.3f} 2010s {r['d_P_2010s']:+.3f}")
json.dump(out, open(f"{ROOT}/results/dsp_components.json", "w"), indent=1); open(f"{ROOT}/results/dsp_components.txt", "w").write("\n".join(txt) + "\n"); print("\n".join(txt))
