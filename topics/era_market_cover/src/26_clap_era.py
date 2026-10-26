"""Independent zero-shot era evaluator: LAION-CLAP (laion/clap-htsat-unfused, transformers) scores audio against decade text prompts; no PA1 training involved.
audio = accompaniment-like signal resampled to 48 kHz, embeddings averaged over three 10 s chunks; decade probabilities = softmax over 6 decades of the cosine similarities x 100, averaged over 3 prompt templates.
(1) positive control: real accompaniment and mix of the 132 validation clips of dataset A (zero-shot accuracy, spearman of expected decade with true decade);
(2) generations in GENDIR (neutral/1960s/1980s/2010s x seeds) and the real 24 songs: P(decade). usage: python 26_clap_era.py --out results/clap_era.json"""
import argparse, json, os, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import ClapModel, ClapProcessor
from common import ROOT, SR, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
mid = "laion/clap-htsat-unfused"; proc = ClapProcessor.from_pretrained(mid); model = ClapModel.from_pretrained(mid).eval().cuda()
T = ["music from the {}", "a recording made in the {}", "a song from the {} sounding like that decade"]
tx = [t.format(d) for d in LABELS["A"] for t in T]
with torch.no_grad():
    ti = proc(text=tx, return_tensors="pt", padding=True).to("cuda"); te = model.get_text_features(**ti); te = torch.nn.functional.normalize(te, dim=-1).view(6, len(T), -1).mean(1); te = torch.nn.functional.normalize(te, dim=-1)
def probs(y):
    x = AF.resample(torch.from_numpy(np.asarray(y, dtype=np.float32)), SR, 48000).numpy(); n = len(x) // 480000; embs = []
    for k in range(max(n, 1)):
        c = x[k * 480000:(k + 1) * 480000]; inp = proc(audios=c, sampling_rate=48000, return_tensors="pt").to("cuda")
        with torch.no_grad(): embs.append(torch.nn.functional.normalize(model.get_audio_features(**inp), dim=-1))
    ae = torch.nn.functional.normalize(torch.cat(embs).mean(0, keepdim=True), dim=-1); return torch.softmax(100 * (ae @ te.T)[0], 0).cpu().numpy().tolist()
out = {"real_val": [], "gen": []}; rows = manifest("A")
for r in rows:
    if r["split"] != "validation": continue
    z = np.load(os.path.join(ROOT, "stems", "A", r["sample_id"] + ".npz")); out["real_val"].append({"clip": r["sample_id"], "true": r["label"], "acc": probs(z["accomp"].astype(np.float32)), "mix": probs(sf.read(r["path"], dtype="float32")[0])})
print("real val done", len(out["real_val"]), flush=True)
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
for s in ids:
    out["gen"].append({"clip": s, "cond": "identity", "seed": 0, "p": probs(np.load(os.path.join(ROOT, "stems", "A", s + ".npz"))["accomp"].astype(np.float32))})
    out["gen"].append({"clip": s, "cond": "identity_codec", "seed": 0, "p": probs(np.load(os.path.join(ROOT, "stems_codec", "A", s + ".npz"))["accomp"].astype(np.float32))})
    for e in ("neutral", "1960s", "1980s", "2010s"):
        for sd in a.seeds: out["gen"].append({"clip": s, "cond": e, "seed": sd, "p": probs(sf.read(os.path.join(G, f"{s}_{e}_s{sd}_acc.wav"), dtype="float32")[0])})
json.dump(out, open(a.out, "w")); print("done", len(out["gen"]))
