"""Cue ablation on REAL validation stems: which acoustic cues do the trained probes use? Interventions are applied to the HTDemucs stem waveform (24 kHz), loudness-normalized to -23 LUFS, embedded with MERT-v2-30s,
and scored with the probe trained on TRAIN unmodified stems (same layer/C as probe_<ds>.json). Interventions: low-pass (4 k, 2 k, 1 kHz), high-pass (200, 500, 1000 Hz), band-pass 300-3000 Hz, soft-clip compression (tanh drive 4), time-shuffle of segments (2 s, 0.5 s),
time-reversal, pitch shift +-2 semitones, additive white noise at -30 dB re stem RMS, and the unmodified control. Reports val accuracy, paired difference to unmodified with McNemar exact p, mean P(true class), and for A the mean predicted expected decade.
usage: python 57_cue_ablation.py A|B -> results/cue_ablation_<ds>.json"""
import sys, json, os, numpy as np, torch, librosa, pyloudnorm as pyln
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]; proc, model = load_mert(); meter = pyln.Meter(SR)
def norm(v, t=-23.0):
    l = meter.integrated_loudness(v.astype(np.float64)); return np.clip(pyln.normalize.loudness(v.astype(np.float64), l, t), -1, 1).astype(np.float32) if np.isfinite(l) and np.abs(v).max() > 1e-4 else v
def embed(w, L):
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = model(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    return torch.stack([((h * m).sum(1) / m.sum(1).clamp_min(1)) if m is not None else h.mean(1) for h in o.hidden_states]).squeeze(1)[L].float().cpu().numpy()
lp = lambda f: (lambda w: sosfiltfilt(butter(8, f, "low", fs=SR, output="sos"), w)); hp = lambda f: (lambda w: sosfiltfilt(butter(8, f, "high", fs=SR, output="sos"), w))
def shuf(sec):
    def f(w):
        n = int(sec * SR); seg = [w[i:i + n] for i in range(0, len(w), n)]; order = np.random.default_rng(1).permutation(len(seg)); return np.concatenate([seg[i] for i in order])
    return f
def noise(w): r = np.sqrt((w ** 2).mean()); return w + np.random.default_rng(2).standard_normal(len(w)).astype(np.float32) * r * 10 ** (-30 / 20)
IV = {"unmodified": lambda w: w, "lowpass_4k": lp(4000), "lowpass_2k": lp(2000), "lowpass_1k": lp(1000), "highpass_200": hp(200), "highpass_500": hp(500), "highpass_1k": hp(1000),
      "bandpass_300_3000": lambda w: hp(300)(lp(3000)(w)), "compress_tanh4": lambda w: np.tanh(4 * w / (np.abs(w).max() + 1e-9)) , "timeshuffle_2s": shuf(2.0), "timeshuffle_0.5s": shuf(0.5), "time_reverse": lambda w: w[::-1].copy(),
      "pitch_+2": lambda w: librosa.effects.pitch_shift(w, sr=SR, n_steps=2), "pitch_-2": lambda w: librosa.effects.pitch_shift(w, sr=SR, n_steps=-2), "noise_-30dB": noise}
conds = ["acc_norm", "voc_norm"]; out = {"ds": ds, "n_val": len(va), "conds": {}}
for c in conds:
    L, C = P[c]["layer"], P[c]["C"]; X = np.load(os.path.join(ROOT, "emb", f"{ds}_{c}.npy"))[:, L].astype(np.float32); pr = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr])
    key = "accomp" if c.startswith("acc") else "vocals"; stems = {i: np.load(os.path.join(ROOT, "stems", ds, rows[i]["sample_id"] + ".npz"))[key].astype(np.float32) for i in va}; res = {}; base_ok = None
    for name, f in IV.items():
        E = np.stack([embed(norm(f(stems[i]).astype(np.float32)), L) for i in va]); p = pr.predict_proba(E); ok = (p.argmax(1) == y[va]); base_ok = ok if name == "unmodified" else base_ok
        b = int((ok & ~base_ok).sum()); c_ = int((~ok & base_ok).sum()); mp = binomtest(b, b + c_, 0.5).pvalue if b + c_ > 0 else 1.0
        r = {"acc": float(ok.mean()), "delta_vs_unmodified": float(ok.mean() - base_ok.mean()), "mcnemar_only_modified_right": b, "mcnemar_only_unmodified_right": c_, "mcnemar_p": float(mp), "mean_P_true": float(p[np.arange(len(va)), y[va]].mean())}
        if ds == "A": r["mean_expected_decade"] = float((p @ np.arange(len(lab))).mean()); r["mean_true_decade"] = float(y[va].mean())
        res[name] = r; print(c, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    out["conds"][c] = res
json.dump(out, open(os.path.join(ROOT, "results", f"cue_ablation_{ds}.json"), "w"), indent=1)
