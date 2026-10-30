"""E07: Frechet distance of generated sets to real PA1 clips of the target decade, with a song-level bootstrap. Reads the av-benchmark feature caches written by run_avb.sh (avb/cache/pred_<set>_<era>/ and gt_<decade>/; VGGish frames (10 s -> 10 x 128) and PANN embeddings), recomputes FD with the av-benchmark formula (mu/cov of the pooled frames, sqrtm) and checks it against the av-benchmark output json (point estimate must agree).
Statistic per set and decade d: Delta FD = FD(era=d prompt, real train clips of decade d) - FD(neutral prompt, same real clips); negative = the era prompt moves the generated distribution closer to that decade. 95% CI: bootstrap over the 24 validation songs (all 3 seeds of a song resampled together), 1000 resamples; the real reference is fixed (171 clips per decade). Caveat to keep with the numbers: VGGish 128-d with 24 x 3 x 10 = 720 generated frames per condition; FD is biased upward at small samples, the same n is used for era and neutral so the difference is comparable.
usage: python 83_fd_bootstrap.py set1 set2 ...   (run in ~/era_market with venv; writes results/fd_boot_<set>.json)"""
import sys, json, os, numpy as np, torch
from scipy import linalg
W = os.path.expanduser("~/era_market/avb"); B = 1000; rng = np.random.default_rng(0)
def fd(a, b):
    m1, m2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm = linalg.sqrtm(s1.dot(s2), disp=False)[0]
    if not np.isfinite(cm).all(): o = np.eye(len(m1)) * 1e-6; cm = linalg.sqrtm((s1 + o).dot(s2 + o))
    cm = cm.real; d = m1 - m2; return float(d.dot(d) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cm))
def load(path, key):
    d = torch.load(f"{path}/{key}", weights_only=True); out = {}
    for k, v in d.items(): out.setdefault(k.split("_")[0] + "_" + k.split("_")[1], []).append(v.numpy().reshape(-1, v.shape[-1]))   # group clips by song (A_<id>)
    return {s: np.concatenate(v) for s, v in out.items()}
for s in sys.argv[1:]:
    res = {}
    for feat, key, fdname in (("vgg", "vggish_features.pth", "FD-VGG"),):   # FD-PANN (2048-d, one vector per clip) and FD-PaSST are not bootstrapped: 72 generated clips per condition < feature dimension, covariance rank-deficient
        for dec in ("1960s", "1980s", "2010s"):
            ref = np.concatenate([v.reshape(-1, v.shape[-1]) for v in torch.load(f"{W}/cache/gt_{dec}/{key}", weights_only=True).values()])
            G = {e: load(f"{W}/cache/pred_{s}_{e}", key) for e in (dec, "neutral")}; songs = sorted(G[dec]); r = {}
            for e in (dec, "neutral"):
                r[f"fd_{e}"] = fd(np.concatenate([G[e][x] for x in songs]), ref)
                j = f"{W}/out/{s}_{e}_vs_{dec}.json"; r[f"avbench_{e}"] = json.load(open(j))[fdname] if os.path.exists(j) else None
            dl = []
            for _ in range(B):
                idx = rng.choice(len(songs), len(songs)); dl.append(fd(np.concatenate([G[dec][songs[i]] for i in idx]), ref) - fd(np.concatenate([G["neutral"][songs[i]] for i in idx]), ref))
            r["delta"] = r[f"fd_{dec}"] - r["fd_neutral"]; r["ci95"] = [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))]; res[f"{fdname}|{dec}"] = r
            print(s, fdname, dec, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "ci95"}, [round(x, 3) for x in r["ci95"]], flush=True)
    json.dump(res, open(os.path.expanduser(f"~/era_market/results/fd_boot_{s}.json"), "w"), indent=1)
