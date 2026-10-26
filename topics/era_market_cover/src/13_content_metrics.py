"""Content-preservation metrics for the generated remixes (CPU), following SURVEY_metrics.md.
 - song identification: for each generated remix, rank the 12 original mixes by chroma-DTW similarity (chroma_cqt of the signal high-passed at middle C, 261 Hz): top-1, reciprocal rank (chance top-1 = 1/12, chance MRR = 0.258)
 - chord agreement: template-matching major/minor chords (24 classes) per frame from chroma, DTW-aligned agreement with the original mix, minus the mean agreement after rotating the generated labels by 1..11 semitones (rotation floor)
 - folded delta-BPM between original mix and generated remix (librosa beat tempo, octave-folded)
usage: python 13_content_metrics.py --out results/content_metrics.json"""
import argparse, json, os, numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
G = os.path.join(ROOT, os.environ.get("GENDIR", "gen")); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
rows = {r["sample_id"]: r for r in manifest("A")}; ERAS = ["neutral", "1960s", "1980s", "2010s"]; HP = butter(4, 261, btype="high", fs=SR, output="sos")
def chroma(y): return librosa.feature.chroma_cqt(y=sosfilt(HP, y).astype(np.float32), sr=SR, hop_length=1024)
def dtw_sim(A, B):
    D, wp = librosa.sequence.dtw(X=A, Y=B, metric="cosine"); return float(1 - D[-1, -1] / len(wp)), wp[::-1]
T = np.zeros((24, 12)); 
for k in range(12): T[k, [k, (k + 4) % 12, (k + 7) % 12]] = 1; T[12 + k, [k, (k + 3) % 12, (k + 7) % 12]] = 1
def chords(C): Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9); return (T @ Cn).argmax(0)
def rot(lbl, s): return np.where(lbl < 12, (lbl + s) % 12, 12 + (lbl - 12 + s) % 12)
def bpm(y):
    t = librosa.feature.tempo(y=y, sr=SR)[0]; return float(t)
def folded_dbpm(b1, b2):   # distance in BPM after folding b2 by factors of 2 to the octave closest to b1
    return float(min(abs(b1 - b2 * f) for f in (0.25, 0.5, 1, 2, 4)))
O = {}
for s in ids:
    y = sf.read(rows[s]["path"], dtype="float32")[0]; C = chroma(y); O[s] = {"C": C, "ch": chords(C), "bpm": bpm(y)}
out = []
for s in ids:
    for e in ERAS:
        for sd in [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]:
            g = sf.read(os.path.join(G, f"{s}_{e}_s{sd}_remix.wav"), dtype="float32")[0]; Cg = chroma(g); cg = chords(Cg)
            sims = {t: dtw_sim(O[t]["C"], Cg)[0] for t in ids}; order = sorted(ids, key=lambda t: -sims[t]); rank = order.index(s) + 1
            _, wp = dtw_sim(O[s]["C"], Cg); ag = float(np.mean(O[s]["ch"][wp[:, 0]] == cg[wp[:, 1]])); fl = float(np.mean([np.mean(O[s]["ch"][wp[:, 0]] == rot(cg, k)[wp[:, 1]]) for k in range(1, 12)]))
            out.append({"clip": s, "cond": e, "seed": sd, "rank": rank, "top1": int(rank == 1), "rr": 1 / rank, "sim_own": sims[s], "chord_agree": ag, "chord_rotation_floor": fl, "dbpm": folded_dbpm(O[s]["bpm"], bpm(g))})
json.dump(out, open(a.out, "w")); print("saved", len(out))
