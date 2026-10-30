"""Long-term-spectrum-matched control for the generated-audio Deltas (follow-up to 75_bandwidth_check.py).
Each era clip (1960s/1980s/2010s) is equalized so that its long-term average spectrum (STFT n_fft 2048, hop 512, power averaged over time, smoothed over +-8 bins, gain limited to +-24 dB) equals that of the neutral-prompt clip of the same song and seed; the neutral clip is kept as is. The matched clips are scored with the codec-matched probes exactly as in 10_eval_gen.py (loudness normalization, MERT L18, MuQ L9). Delta = per-song mean over seeds of P(target decade | matched era clip) - P(target decade | neutral clip), 95% song-bootstrap CI, next to the unmatched Delta recomputed from the same pipeline.
usage: python 76_spectrum_match.py label=gendir:evaljson ... -> results/spectrum_match.{json,txt}"""
import sys, os, json, numpy as np, soundfile as sf, librosa
from scipy.ndimage import uniform_filter1d
from common import ROOT, SR
import eraprobe as EP
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; rng = np.random.default_rng(0); out = {}; txt = []
probes, layer = EP.load_probes()
def ltas(S): return uniform_filter1d((np.abs(S) ** 2).mean(1), 17, mode="nearest") + 1e-12
def match(y, ref):
    S = librosa.stft(y, n_fft=2048, hop_length=512); g = np.clip(np.sqrt(ltas(librosa.stft(ref, n_fft=2048, hop_length=512)) / ltas(S)), 10 ** (-24 / 20), 10 ** (24 / 20))
    return librosa.istft(S * g[:, None], hop_length=512, length=len(y)).astype(np.float32)
for arg in sys.argv[1:]:
    label, rest = arg.split("=", 1); gd, ev = rest.split(":"); G = os.path.join(ROOT, gd); R = json.load(open(os.path.join(ROOT, ev)))["rows"]
    songs = sorted({r["clip"] for r in R}); seeds = sorted({r["seed"] for r in R if r["cond"] == "neutral"}); waves = {}
    for s in songs:
        for sd in seeds:
            rd = lambda e: sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0]; n0 = rd("neutral"); waves[f"{s}|neutral|{sd}"] = n0
            for e in TD: waves[f"{s}|{e}|{sd}"] = rd(e); waves[f"{s}|{e}m|{sd}"] = match(rd(e), n0)
    emb = EP.embed_waves(waves, layer); pr = EP.predict(probes, emb); res = {}; line = [label]
    for enc in ("mert", "muq"):
        for tag in ("", "m"):
            cells = []
            for e, t in TD.items():
                P = lambda c: np.array([np.mean([pr[enc][f"{s}|{c}|{sd}"][t] for sd in seeds]) for s in songs]); d = P(e + tag) - P("neutral"); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                res[f"{enc}_{'matched' if tag else 'orig'}_{e}"] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "pos": int((d > 0).sum()), "n": len(d)}
                cells.append(f"{e} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
            line.append(f"  {enc:4s} {'spectrum-matched' if tag else 'unmatched       '}: " + " | ".join(cells))
    out[label] = res; txt += line; print("\n".join(line), flush=True)
json.dump(out, open(f"{ROOT}/results/spectrum_match.json", "w"), indent=1); open(f"{ROOT}/results/spectrum_match.txt", "w").write("\n".join(txt) + "\n")
