"""Song-identity metric from a cover-song-identification embedding: Discogs-VINet (Araz et al. 2024, ISMIR2024 branch checkpoint; code and weights are AGPLv3/third party and are NOT vendored in this repo).
Embedding = whole 30 s clip, CQT as in the authors' inference code (22.05 kHz, hop 512, 84 bins, magnitude, pad to context 8000 frames, mean-downsample x20, scale by max), L2-normalized 'CQTNet' output.
For every generated remix (original vocal + generated accompaniment) in a generation directory: cosine similarity to the embedding of its own original mix, and retrieval among all original mixes of the 24 evaluation songs (top-1, MRR; chance 1/24).
Reference rows: real mix re-embedded, and 'recomb' = original vocal + original accompaniment. usage: python 54_vinet_identity.py gen gen_lora gen_lora_v2 ... -> results/vinet_identity.json
Needs the repo cloned at ~/era_market/ext/Discogs-VINet (branch ISMIR2024)."""
import sys, os, re, json, glob, numpy as np, torch, librosa, soundfile as sf
from scipy.stats import wilcoxon
from collections import defaultdict
from common import ROOT, SR, manifest
sys.path.insert(0, os.path.expanduser("~/era_market/ext/Discogs-VINet")); from model.nets.cqtnet import CQTNet
net = CQTNet(); sd = torch.load(os.path.expanduser("~/era_market/ext/Discogs-VINet/logs/checkpoints/Discogs-VINet/model_checkpoint.pth"), map_location="cpu", weights_only=True)["model_state_dict"]; net.load_state_dict(sd); net = net.cuda().eval()
def emb(y24):
    y = librosa.resample(np.asarray(y24, dtype=np.float32), orig_sr=SR, target_sr=22050); c = np.abs(librosa.cqt(y=y, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12)).T.astype(np.float16).astype(np.float32)
    if c.shape[0] < 8000: c = np.pad(c, ((0, 8000 - c.shape[0]), (0, 0)))
    T = c.shape[0] // 20; c = c[:T * 20].reshape(T, 20, 84).mean(1); c = np.where(c < 0, 0, c); c /= c.max() + 1e-6
    with torch.inference_mode(): return net(torch.as_tensor(c.T)[None, None].cuda()).float().cpu().numpy()[0]
rows = {r["sample_id"]: r for r in manifest("A")}; ids = [l.split(",")[0] for l in open(os.path.join(ROOT, "gen", "picked.txt")).read().split("\n")]
O = np.stack([emb(sf.read(rows[s]["path"], dtype="float32")[0]) for s in ids]); res = {"n_songs": len(ids), "chance_top1": 1 / len(ids)}
def score(e, s):
    sims = O @ e; rank = int(np.argsort(-sims).tolist().index(ids.index(s))) + 1; return float(sims[ids.index(s)]), int(rank == 1), 1.0 / rank
ref = [score(emb(sf.read(rows[s]["path"], dtype="float32")[0]), s) for s in ids]; res["real_mix_reembedded"] = {"cos": float(np.mean([r[0] for r in ref])), "top1": float(np.mean([r[1] for r in ref]))}
rc = []
for s in ids:
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); rc.append(score(emb(z["vocals"].astype(np.float32) + z["accomp"].astype(np.float32)), s))
res["recomb_original_stems"] = {"cos": float(np.mean([r[0] for r in rc])), "top1": float(np.mean([r[1] for r in rc]))}
OUT = os.path.join(ROOT, "results", "vinet_identity.json"); old = json.load(open(OUT)) if os.path.exists(OUT) else {}; res["dirs"] = old.get("dirs", {})   # merge with earlier runs
for d in sys.argv[1:]:
    fs = sorted(glob.glob(os.path.join(ROOT, d, "*_remix.wav"))); R = defaultdict(list)
    for f in fs:
        m = re.match(r"^(A_[0-9a-f]+)_(.+)_s(\d+)_remix\.wav$", os.path.basename(f))
        if not m or m.group(1) not in ids: continue
        s, era, sd_ = m.group(1), m.group(2), int(m.group(3)); R[era].append((s, sd_) + score(emb(sf.read(f, dtype="float32")[0]), s))
    if not R: continue
    out = {}
    for era, v in R.items(): out[era] = {"n": len(v), "cos": float(np.mean([x[2] for x in v])), "top1": float(np.mean([x[3] for x in v])), "mrr": float(np.mean([x[4] for x in v]))}
    if "neutral" in R:
        sc = lambda era: {s: np.mean([x[2] for x in R[era] if x[0] == s]) for s in {x[0] for x in R[era]}}
        n0 = sc("neutral")
        for era in R:
            if era == "neutral": continue
            e = sc(era); ss = sorted(set(n0) & set(e)); dlt = np.array([e[s] - n0[s] for s in ss])
            if len(ss) > 5 and np.any(dlt != 0): out[era]["delta_cos_vs_neutral"] = {"mean": float(dlt.mean()), "n_songs": len(ss), "wilcoxon_p": float(wilcoxon(dlt).pvalue)}
    res["dirs"][d] = out; print(d, json.dumps({e: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "delta_cos_vs_neutral"} for e, v in out.items()}), flush=True)
json.dump(res, open(os.path.join(ROOT, "results", "vinet_identity.json"), "w"), indent=1); print("real", res["real_mix_reembedded"], "recomb", res["recomb_original_stems"])
