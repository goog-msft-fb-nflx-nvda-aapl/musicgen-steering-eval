"""Cue ablation with RETRAINED probes ("information left", not distribution shift): for each intervention, every labeled clip's stem (accompaniment or vocals; HTDemucs) is modified, loudness-normalized, embedded with MERT-v2-30s,
a probe is re-fitted on the modified TRAIN clips (same layer and C as probe_<ds>.json, StandardScaler + logistic regression) and scored on the modified validation clips (bootstrap 95% CI, McNemar vs the unmodified-trained control row).
Interventions as in 57_cue_ablation.py minus pitch shift (cost). Embeddings cached in emb_cue/. usage: python 57b_cue_retrain.py A|B"""
import sys, json, os, numpy as np, torch, pyloudnorm as pyln
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
ds = sys.argv[1]; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]; lab_idx = np.where(y >= 0)[0]
P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]; proc, model = load_mert(); meter = pyln.Meter(SR); os.makedirs(os.path.join(ROOT, "emb_cue"), exist_ok=True)
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
IV = {"unmodified": lambda w: w, "lowpass_2k": lp(2000), "highpass_500": hp(500), "bandpass_300_3000": lambda w: hp(300)(lp(3000)(w)), "compress_tanh4": lambda w: np.tanh(4 * w / (np.abs(w).max() + 1e-9)),
      "timeshuffle_2s": shuf(2.0), "timeshuffle_0.5s": shuf(0.5), "time_reverse": lambda w: w[::-1].copy(), "noise_-30dB": noise}
out = {"ds": ds, "n_train": len(tr), "n_val": len(va), "conds": {}}
for c in ("acc_norm", "voc_norm"):
    L, C = P[c]["layer"], P[c]["C"]; key = "accomp" if c.startswith("acc") else "vocals"; res = {}; base_ok = None
    for name, f in IV.items():
        cache = os.path.join(ROOT, "emb_cue", f"{ds}_{c}_{name}_L{L}.npy")
        if os.path.exists(cache): E = np.load(cache)
        else:
            E = np.zeros((len(rows), 1024), np.float32)
            for n, i in enumerate(lab_idx):
                E[i] = embed(norm(f(np.load(os.path.join(ROOT, "stems", ds, rows[i]["sample_id"] + ".npz"))[key].astype(np.float32)).astype(np.float32)), L)
                if n % 300 == 0: print(c, name, n, len(lab_idx), flush=True)
            np.save(cache, E)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(E[tr], y[tr]); ok = (m.predict(E[va]) == y[va]); base_ok = ok if name == "unmodified" else base_ok
        bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(2000)]; b = int((ok & ~base_ok).sum()); c_ = int((~ok & base_ok).sum())
        res[name] = {"acc": float(ok.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "delta_vs_unmodified": float(ok.mean() - base_ok.mean()), "only_modified_right": b, "only_unmodified_right": c_, "mcnemar_p": float(binomtest(b, b + c_, 0.5).pvalue) if b + c_ else 1.0}
        print(c, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items() if k != "ci95"}, flush=True)
    out["conds"][c] = res
json.dump(out, open(os.path.join(ROOT, "results", f"cue_retrain_{ds}.json"), "w"), indent=1)
