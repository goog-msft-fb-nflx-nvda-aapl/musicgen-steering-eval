"""Shuffle-and-remix test (after Hsieh et al. 2020, arXiv 2002.06817): which stem does the mix probe follow when the vocal of one clip is combined with the accompaniment of a clip with a DIFFERENT label?
Validation clips only (probe trained on TRAIN mix_norm, layer and C from probe_<ds>.json). For each validation clip i (vocal source) K accompaniment sources j with label_j != label_i (seeded), recombined = norm(vocal_i + accomp_j) to -23 LUFS,
embedded with MERT-v2-30s, scored with the TRAIN mix probe. Reports per pair: predicted label equals vocal label / accompaniment label / other, mean P(vocal label) vs P(accompaniment label), bootstrap CI over vocal-source clips.
For dataset A also regresses the expected decade on the vocal-source decade and the accompaniment-source decade. Reference rows: same-clip recombination (vocal_i + accomp_i, emb recomb_norm).
usage: python 50_shuffle_remix.py A|B [K]   -> results/shuffle_remix_<ds>.json"""
import sys, json, os, numpy as np, torch, pyloudnorm as pyln
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, SR, LABELS, manifest
from mert_util import load_mert
ds = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 3; lab = LABELS[ds]; rows = manifest(ds); rng = np.random.default_rng(0)
y = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); split = np.array([r["split"] for r in rows]); P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]["mix_norm"]
X = np.load(os.path.join(ROOT, "emb", f"{ds}_mix_norm.npy"))[:, P["layer"]].astype(np.float32); tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]
probe = make_pipeline(StandardScaler(), LogisticRegression(C=P["C"], max_iter=3000)).fit(X[tr], y[tr])
print("probe layer", P["layer"], "C", P["C"], "val acc on real mixes", float((probe.predict(X[va]) == y[va]).mean()), flush=True)
Xr = np.load(os.path.join(ROOT, "emb", f"{ds}_recomb_norm.npy"))[:, P["layer"]].astype(np.float32); print("val acc on same-clip recombination", float((probe.predict(Xr[va]) == y[va]).mean()), flush=True)
pairs = []
for i in va:
    cand = [j for j in va if y[j] != y[i]]
    for j in rng.choice(cand, K, replace=False): pairs.append((int(i), int(j)))
proc, model = load_mert(); meter = pyln.Meter(SR)
def norm(v, target=-23.0):
    l = meter.integrated_loudness(v.astype(np.float64)); return np.clip(pyln.normalize.loudness(v.astype(np.float64), l, target), -1, 1).astype(np.float32) if np.isfinite(l) and np.abs(v).max() > 1e-4 else v
stem = lambda k: np.load(os.path.join(ROOT, "stems", ds, rows[k]["sample_id"] + ".npz"))
Z = {}; E = []
for n, (i, j) in enumerate(pairs):
    for k in (i, j):
        if k not in Z: z = stem(k); Z[k] = (z["vocals"].astype(np.float32), z["accomp"].astype(np.float32))
    n_ = min(len(Z[i][0]), len(Z[j][1])); w = norm(Z[i][0][:n_] + Z[j][1][:n_])
    inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): o = model(**inp, output_hidden_states=True)
    m = o.feature_attention_mask[..., None].float() if getattr(o, "feature_attention_mask", None) is not None else None
    hs = torch.stack([((h * m).sum(1) / m.sum(1).clamp_min(1)) if m is not None else h.mean(1) for h in o.hidden_states]).squeeze(1); E.append(hs[P["layer"]].float().cpu().numpy())
    if n % 100 == 0: print(n, len(pairs), flush=True)
E = np.array(E); pr = probe.predict_proba(E); vi = np.array([p[0] for p in pairs]); aj = np.array([p[1] for p in pairs]); lv, la = y[vi], y[aj]; pred = pr.argmax(1)
res = {"ds": ds, "n_pairs": len(pairs), "K": K, "probe_layer": P["layer"], "val_acc_real_mix": float((probe.predict(X[va]) == y[va]).mean()), "val_acc_same_clip_recomb": float((probe.predict(Xr[va]) == y[va]).mean())}
res["pred_eq_vocal_label"] = float((pred == lv).mean()); res["pred_eq_accomp_label"] = float((pred == la).mean()); res["pred_other"] = float(((pred != lv) & (pred != la)).mean())
pv = pr[np.arange(len(pairs)), lv]; pa = pr[np.arange(len(pairs)), la]; res["mean_P_vocal_label"] = float(pv.mean()); res["mean_P_accomp_label"] = float(pa.mean())
clips = np.unique(vi); d = pv - pa
def boot(f, B=5000):
    bs = [f(rng.choice(clips, len(clips))) for _ in range(B)]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
byc = {c: np.where(vi == c)[0] for c in clips}; stat = lambda cs: float(np.mean(np.concatenate([d[byc[c]] for c in cs])))
res["mean_P_vocal_minus_accomp"] = float(d.mean()); res["ci95_cluster_bootstrap_over_vocal_clips"] = boot(stat)
eqv = lambda cs: float(np.mean(np.concatenate([(pred == lv)[byc[c]] for c in cs]))); eqa = lambda cs: float(np.mean(np.concatenate([(pred == la)[byc[c]] for c in cs])))
res["ci95_pred_eq_vocal"] = boot(eqv); res["ci95_pred_eq_accomp"] = boot(eqa)
if ds == "A":
    ed = pr @ np.arange(len(lab)); A = np.stack([np.ones(len(pairs)), lv, la], 1); beta = np.linalg.lstsq(A, ed, rcond=None)[0]; res["expected_decade_regression"] = {"intercept": float(beta[0]), "beta_vocal_source_decade": float(beta[1]), "beta_accomp_source_decade": float(beta[2])}
    def reg(cs):
        idx = np.concatenate([byc[c] for c in cs]); b = np.linalg.lstsq(A[idx], ed[idx], rcond=None)[0]; return float(b[2] - b[1])
    res["accomp_minus_vocal_beta"] = reg(clips); res["ci95_accomp_minus_vocal_beta"] = boot(reg)
res["per_vocal_label"] = {lab[k]: {"pred_eq_vocal": float((pred == lv)[lv == k].mean()), "pred_eq_accomp": float((pred == la)[lv == k].mean()), "n": int((lv == k).sum())} for k in range(len(lab))}
json.dump(res, open(os.path.join(ROOT, "results", f"shuffle_remix_{ds}.json"), "w"), indent=1); print(json.dumps(res, indent=1))
