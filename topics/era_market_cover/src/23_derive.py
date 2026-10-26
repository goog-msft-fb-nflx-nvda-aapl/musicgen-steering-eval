"""CPU-derived signals. (1) HPSS split of the accompaniment stem (librosa.effects.hpss, margin 1) -> stems_derived/{harm,perc}/<ds>/<id>.npz (x, fp16), all clips of A and B.
(2) parametric production-only control: the REAL accompaniment of the 24 evaluation songs processed by three hand-built pedalboard chains (not generated audio) written like a generation folder
gen_fx/<id>_<era>_s0_{acc,remix}.wav (neutral = unprocessed) so 10_eval_gen.py / 13_content_metrics.py can score them with the same probes.  chains are our own approximations of era production, not validated.
usage: python 23_derive.py hpss | fx"""
import sys, os, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool
from common import ROOT, SR, manifest
def hp(args):
    ds, sid = args; z = np.load(os.path.join(ROOT, "stems", ds, sid + ".npz")); y = z["accomp"].astype(np.float32)
    h, p = librosa.effects.hpss(y, margin=1.0)
    for k, v in (("harm", h), ("perc", p)):
        d = os.path.join(ROOT, "stems_derived", k, ds); os.makedirs(d, exist_ok=True); np.savez(os.path.join(d, sid + ".npz"), x=v.astype(np.float16))
def chains():
    from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Distortion, Reverb, Compressor, Limiter, HighShelfFilter, LowShelfFilter, Chorus, Gain
    return {"1960s": Pedalboard([HighpassFilter(80), LowpassFilter(8000), Distortion(drive_db=6), Reverb(room_size=0.25, wet_level=0.12, dry_level=0.9), Gain(-1)]),
            "1980s": Pedalboard([Compressor(threshold_db=-18, ratio=3), HighShelfFilter(cutoff_frequency_hz=6000, gain_db=4), Chorus(rate_hz=0.8, depth=0.25, mix=0.25), Reverb(room_size=0.7, wet_level=0.25, dry_level=0.8)]),
            "2010s": Pedalboard([LowShelfFilter(cutoff_frequency_hz=90, gain_db=4), Compressor(threshold_db=-22, ratio=6, attack_ms=10, release_ms=80), HighShelfFilter(cutoff_frequency_hz=8000, gain_db=3), Limiter(threshold_db=-1.5)])}
if __name__ == "__main__":
    if sys.argv[1] == "hpss":
        for ds in ("A", "B"):
            with Pool(24) as P: P.map(hp, [(ds, r["sample_id"]) for r in manifest(ds)], chunksize=8)
            print("hpss done", ds, flush=True)
    else:
        picked = open(os.path.join(ROOT, "gen", "picked.txt")).read(); d = os.path.join(ROOT, "gen_fx"); os.makedirs(d, exist_ok=True); open(os.path.join(d, "picked.txt"), "w").write(picked); C = chains()
        for line in picked.split("\n"):
            s = line.split(",")[0]; z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
            for era in ("neutral", "1960s", "1980s", "2010s"):
                y = acc if era == "neutral" else C[era](acc[None], SR)[0]
                y = y * (np.sqrt((acc ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))
                sf.write(os.path.join(d, f"{s}_{era}_s0_acc.wav"), y, SR); sf.write(os.path.join(d, f"{s}_{era}_s0_remix.wav"), np.clip(voc + y, -1, 1), SR)
        print("fx done")
