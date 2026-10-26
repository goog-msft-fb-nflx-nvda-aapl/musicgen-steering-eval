"""Remix-level ablation: rebuild remixes of the MusicGen g=3 seed-0 generations with the generated accompaniment scaled by -12, -6, 0, +6 dB relative to its default level (RMS of the original accompaniment), vocal stem unchanged,
into gen_lvl_<tag>/ (+ picked.txt) so that 13_content_metrics.py can score them (GENDIR=gen_lvl_m6 SEEDS=0 ...). usage: python 38_remix_level.py"""
import os, numpy as np, soundfile as sf
from common import ROOT, SR
ids = [l.split(",")[0] for l in open(os.path.join(ROOT, "gen", "picked.txt")).read().split("\n")]
for tag, db in (("m12", -12), ("m6", -6), ("p0", 0), ("p6", 6)):
    d = os.path.join(ROOT, f"gen_lvl_{tag}"); os.makedirs(d, exist_ok=True); open(os.path.join(d, "picked.txt"), "w").write(open(os.path.join(ROOT, "gen", "picked.txt")).read())
    for s in ids:
        voc = np.load(os.path.join(ROOT, "stems", "A", s + ".npz"))["vocals"].astype(np.float32)
        for e in ("neutral", "1960s", "1980s", "2010s"):
            g = sf.read(os.path.join(ROOT, "gen", f"{s}_{e}_s0_acc.wav"), dtype="float32")[0] * 10 ** (db / 20); sf.write(os.path.join(d, f"{s}_{e}_s0_remix.wav"), np.clip(voc + g, -1, 1), SR)
print("done")
