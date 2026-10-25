"""Shared helpers: loudness normalization, MERT-v2 and MuQ embedding of a dict of waveforms (MERT first, MuQ imported afterwards), decade probes trained on dataset-A TRAIN stems."""
import json, os, numpy as np, torch, pyloudnorm as pyln
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import ROOT, SR, manifest, LABELS
_meter = pyln.Meter(SR)
def norm(y):
    y = y.astype(np.float32)
    if np.abs(y).max() < 1e-4: return y
    l = _meter.integrated_loudness(y.astype(np.float64)); return y if not np.isfinite(l) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
def load_probes(train_cond="acccodec_norm", probe_mert="probe_codec_mert_A.json", probe_muq="probe_codec_muq_A.json"):
    pj = {"mert": json.load(open(os.path.join(ROOT, "results", probe_mert))), "muq": json.load(open(os.path.join(ROOT, "results", probe_muq)))}
    layer = {k: pj[k]["conds"][train_cond]["layer"] for k in pj}
    rows = manifest("A"); y = np.array([LABELS["A"].index(r["label"]) if r["label"] else -1 for r in rows]); tr = np.where(np.array([r["split"] for r in rows]) == "train")[0]
    probes = {}
    for k, pre in (("mert", ""), ("muq", "muq_")):
        E = np.load(os.path.join(ROOT, "emb", f"A_{pre}{train_cond}.npy")).astype(np.float32)[:, layer[k]]
        probes[k] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(E[tr], y[tr])
    return probes, layer
def embed_waves(waves, layer):
    """waves: dict name -> float32 24 kHz mono. returns {'mert': {name: vec}, 'muq': {name: vec}}"""
    from transformers import AutoFeatureExtractor, AutoModel
    out = {"mert": {}, "muq": {}}; W = {n: norm(w) for n, w in waves.items()}
    proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True); mert = AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True).eval().cuda()
    for n, w in W.items():
        inp = proc(w, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
        with torch.inference_mode(): o = mert(**inp, output_hidden_states=True)
        m = o.feature_attention_mask[..., None].float(); h = o.hidden_states[layer["mert"]]; out["mert"][n] = ((h * m).sum(1) / m.sum(1)).squeeze(0).cpu().numpy()
    del mert; torch.cuda.empty_cache()
    from muq_util import load_muq
    muq = load_muq()
    for n, w in W.items():
        with torch.inference_mode(): oq = muq(torch.from_numpy(w)[None].cuda(), output_hidden_states=True)
        out["muq"][n] = oq["hidden_states"][layer["muq"]].float().mean(1).squeeze(0).cpu().numpy()
    for k in out:
        for n, v in out[k].items(): assert np.isfinite(v).all(), f"non-finite {k} {n}"
    return out
def predict(probes, emb):
    return {k: {n: probes[k].predict_proba(v[None])[0] for n, v in emb[k].items()} for k in emb}
