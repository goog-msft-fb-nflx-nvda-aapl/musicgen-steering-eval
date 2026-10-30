"""Bandwidth/noise-floor check of generated accompaniment (follow-up to 74_dsp_components.py: band-limit and hiss move the probes).
Per generated accompaniment clip (GENDIR/<song>_<era>_s<seed>_acc.wav): loudness-normalized (-23 LUFS) 95% spectral rolloff (Hz) and the energy fraction above 7 kHz (dB); per-song change versus the neutral prompt (seed mean); Spearman correlation across song x era of the change in each feature with the change in the probe's P(target decade) (eval json, MERT / MuQ).
usage: python 75_bandwidth_check.py label=gendir:evaljson ... -> results/bandwidth_check.{json,txt}"""
import sys, os, json, numpy as np, soundfile as sf, librosa
from scipy.stats import spearmanr, wilcoxon
from common import ROOT, SR
from eraprobe import norm
TD = {"1960s": 0, "1980s": 2, "2010s": 5}; out = {}; txt = []
def feats(y):
    y = norm(y); S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2; f = librosa.fft_frequencies(sr=SR, n_fft=2048); tot = S.sum() + 1e-12; c = np.cumsum(S.sum(1)) / tot
    return float(f[np.searchsorted(c, 0.95)]), float(10 * np.log10(S[f > 7000].sum() / tot + 1e-12))
for arg in sys.argv[1:]:
    label, rest = arg.split("=", 1); gd, ev = rest.split(":"); G = os.path.join(ROOT, gd); R = json.load(open(os.path.join(ROOT, ev)))["rows"]
    songs = sorted({r["clip"] for r in R}); seeds = sorted({r["seed"] for r in R if r["cond"] == "neutral"}); eras = ["neutral", "1960s", "1980s", "2010s"]
    F = {}
    for s in songs:
        for e in eras:
            v = np.array([feats(sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0]) for sd in seeds]); F[(s, e)] = v.mean(0)
    res = {"mean": {e: {"rolloff_hz": float(np.mean([F[(s, e)][0] for s in songs])), "hf_db": float(np.mean([F[(s, e)][1] for s in songs]))} for e in eras}, "vs_neutral": {}, "spearman": {}}
    line = [f"{label}: mean rolloff95 (Hz) / energy above 7 kHz (dB): " + "; ".join(f"{e} {res['mean'][e]['rolloff_hz']:.0f} / {res['mean'][e]['hf_db']:.1f}" for e in eras)]
    for e in eras[1:]:
        dr = np.array([F[(s, e)][0] - F[(s, "neutral")][0] for s in songs]); dh = np.array([F[(s, e)][1] - F[(s, "neutral")][1] for s in songs])
        res["vs_neutral"][e] = {"d_rolloff_hz": float(dr.mean()), "p_rolloff": float(wilcoxon(dr).pvalue), "d_hf_db": float(dh.mean()), "p_hf": float(wilcoxon(dh).pvalue)}
        line.append(f"  {e} - neutral: rolloff {dr.mean():+.0f} Hz (p={wilcoxon(dr).pvalue:.4f}), HF {dh.mean():+.2f} dB (p={wilcoxon(dh).pvalue:.4f})")
    for enc in ("mert", "muq"):
        dp, dr, dh = [], [], []
        for e, t in TD.items():
            for s in songs:
                P = lambda c: np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == s and r["cond"] == c]); dp.append(P(e) - P("neutral")); dr.append(F[(s, e)][0] - F[(s, "neutral")][0]); dh.append(F[(s, e)][1] - F[(s, "neutral")][1])
        a, b = spearmanr(dp, dr), spearmanr(dp, dh); res["spearman"][enc] = {"rolloff": [float(a[0]), float(a[1])], "hf": [float(b[0]), float(b[1])], "n": len(dp)}
        line.append(f"  {enc}: Spearman(d P(target), d rolloff) {a[0]:+.2f} (p={a[1]:.3g}); (d P(target), d HF dB) {b[0]:+.2f} (p={b[1]:.3g}); n={len(dp)}")
    out[label] = res; txt += line; print("\n".join(line), flush=True)
json.dump(out, open(f"{ROOT}/results/bandwidth_check.json", "w"), indent=1); open(f"{ROOT}/results/bandwidth_check.txt", "w").write("\n".join(txt) + "\n")
