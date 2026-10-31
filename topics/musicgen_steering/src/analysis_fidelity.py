"""fidelity analysis of arms_fid/*.npz (07_fidelity.py): per method x arm x eta, pooled over 12 notes (2 gen seeds x 20 prompts each = 480 pairs/cell).
metrics vs the unsteered clip of the same (note, seed, prompt): hit rate, clap text-audio score (and paired wilcoxon delta),
paired clap-embedding cosine distance (1 - cos to the unsteered twin), unbiased MMD^2 (rbf, median bandwidth) steered-vs-unsteered embeddings.
controls (random, shuffled; 08_controls.py *_ctl.npz) are included if present in the same dir.
usage: python analysis_fidelity.py <dir with npz> <out json> [out png]
"""
import sys, glob, json, os, numpy as np
from scipy.stats import wilcoxon
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]; METH = ["rfm", "meandiff", "logistic", "lda", "random", "shuffled"]
SUF = lambda m: "ctl" if m in ("random", "shuffled") else "fid"
d, outj = sys.argv[1], sys.argv[2]; outp = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None; MODEL = sys.argv[4] if len(sys.argv) > 4 else "small"  # usage: ... <out json> [out png|-] [small|medium]

def mmd2(X, Y):  # unbiased, rbf, bandwidth = median pairwise dist of pooled
    Z = np.vstack([X, Y]); D = ((Z[:, None] - Z[None]) ** 2).sum(-1); s2 = np.median(D[np.triu_indices(len(Z), 1)]) / 2
    K = np.exp(-D / (2 * s2)); n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())

res = {}
for m in METH:
    if not os.path.exists(f"{d}/musicgen_{MODEL}_C_{m}_{SUF(m)}.npz"): continue
    cells = {}
    for n in NOTES:
        z = np.load(f"{d}/musicgen_{MODEL}_{n.replace('#', 'sharp')}_{m}_{SUF(m)}.npz"); E = z["emb"].astype(np.float32); R = json.loads(str(z["rows"]))
        for i, r in enumerate(R): cells.setdefault((r["arm"], r["eta"]), []).append((n, r, E[i]))
    base = {(n, r["seed"], r["prompt"]): (r, e) for n, r, e in cells[("base", 0.0)]}
    for (arm, eta), L in sorted(cells.items()):
        if arm == "base": continue
        hit = np.mean([r["hit"] for _, r, _ in L]); frac = np.mean([r["frac"] for _, r, _ in L])
        pb = [base[(n, r["seed"], r["prompt"])] for n, r, _ in L]
        dc = np.array([r["clap"] - b[0]["clap"] for (n, r, _), b in zip(L, pb)])
        cd = np.array([1 - float(e @ b[1] / (np.linalg.norm(e) * np.linalg.norm(b[1]))) for (_, _, e), b in zip(L, pb)])
        X = np.stack([e for _, _, e in L]); Y = np.stack([b[1] for b in pb])
        try: p = float(wilcoxon(dc).pvalue)
        except ValueError: p = 1.0
        res[f"{m}|{arm}|{eta}"] = dict(method=m, arm=arm, eta=eta, n=len(L), hit=float(hit), frac=float(frac), clap=float(np.mean([r["clap"] for _, r, _ in L])),
            clap_base=float(np.mean([b[0]["clap"] for b in pb])), dclap=float(dc.mean()), dclap_p=p, twin_cosdist=float(cd.mean()), mmd2=mmd2(X, Y))
json.dump(res, open(outj, "w"), indent=1)
print(f"{'method':9s} {'arm':9s} eta   hit   frac  dCLAP    p       twinDist  MMD2")
for k, v in res.items(): print(f"{v['method']:9s} {v['arm']:9s} {v['eta']:.2f} {v['hit']:.3f} {v['frac']:.3f} {v['dclap']:+.4f} {v['dclap_p']:.1e} {v['twin_cosdist']:.4f}   {v['mmd2']:+.4f}")
if outp:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); col = dict(rfm="#4c78a8", meandiff="#e45756", logistic="#54a24b", lda="#b279a2", random="#7f7f7f", shuffled="#bcbd22")
    for j, (key, lab) in enumerate([("dclap", "mean CLAP text-audio change vs unsteered"), ("twin_cosdist", "CLAP cosine dist to unsteered twin"), ("mmd2", "MMD^2 vs unsteered")]):
        for m in METH:
            for arm, mk in [("const8", "o"), ("musicrfm", "s")]:
                V = sorted([v for v in res.values() if v["method"] == m and v["arm"] == arm], key=lambda v: v["eta"])
                ax[j].plot([v[key] for v in V], [v["hit"] for v in V], mk + "-", c=col[m], label=f"{m} {arm}", ms=5, alpha=.85)
        ax[j].set_xlabel(lab); ax[j].set_ylabel("pooled target-note hit rate"); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.suptitle("musicgen-small, 12 notes pooled, 480 pairs/cell, eta .15-.6 along each line; chroma hit vs clap-htsat-fused fidelity", fontsize=9)
    fig.tight_layout(); fig.savefig(outp, dpi=130)
