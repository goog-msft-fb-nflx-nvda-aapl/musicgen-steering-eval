"""Signal-statistics features per clip (shortcut baseline): integrated loudness, short-term loudness range, crest factor, RMS,
bandwidth (highest frequency whose mean power is within 60 dB of the peak bin), spectral centroid/rolloff-85, zero-crossing rate, noise floor (10th percentile frame RMS dB).
computed on mix and on each stem. saves emb/<ds>_stats_<src>.npy  (N, F).  usage: python 03_stats.py --ds A"""
import argparse, os, numpy as np, soundfile as sf, librosa, pyloudnorm as pyln
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); a = ap.parse_args()
meter = pyln.Meter(SR)
NAMES = ["lufs", "loud_range", "crest_db", "rms_db", "bandwidth_hz", "centroid_hz", "rolloff85_hz", "zcr", "noise_floor_db"]
def feats(y):
    y = y.astype(np.float32); eps = 1e-9
    rms = np.sqrt((y ** 2).mean() + eps); peak = np.abs(y).max() + eps
    try: lufs = meter.integrated_loudness(y.astype(np.float64))
    except Exception: lufs = -70.0
    fr = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]; db = 20 * np.log10(fr + eps)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2; P = S.mean(1); f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    pdb = 10 * np.log10(P + eps); bw = f[np.where(pdb > pdb.max() - 60)[0].max()]
    cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR).mean(); roll = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=SR, roll_percent=.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    return [lufs, np.percentile(db, 95) - np.percentile(db, 10), 20 * np.log10(peak / rms), 20 * np.log10(rms), bw, cen, roll, zcr, np.percentile(db, 10)]
rows = manifest(a.ds); out = {s: [] for s in ("mix", "voc", "acc")}
for r in rows:
    out["mix"].append(feats(sf.read(r["path"], dtype="float32")[0]))
    z = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz"))
    out["voc"].append(feats(z["vocals"].astype(np.float32))); out["acc"].append(feats(z["accomp"].astype(np.float32)))
for s, v in out.items(): np.save(os.path.join(ROOT, "emb", f"{a.ds}_stats_{s}.npy"), np.array(v, dtype=np.float32))
print("saved stats", a.ds, NAMES)
