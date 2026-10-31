"""target -> predicted note confusion for steered clips (small, const8+musicrfm pooled, eta .3/.6), from arms_eval2b (ch_pred, mert_pred).
usage: python analysis_confusion.py results/ablation/arms_eval2b"""
import numpy as np, json, glob, sys, collections, os
N = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
d = sys.argv[1]
for meth in ["meandiff", "rfm"]:
    R = {}
    for f in sorted(glob.glob(f"{d}/musicgen_small_*_{meth}_eval2.npz")) if meth == "meandiff" else [g for g in sorted(glob.glob(f"{d}/musicgen_small_*_eval2.npz")) if "meandiff" not in g]:
        t = os.path.basename(f).split("_")[2].replace("sharp", "#")
        R[t] = json.loads(str(np.load(f, allow_pickle=True)["rows"]))
    print(f"\n== {meth}: per target note, steered (eta>=.3) hit rate and top wrong prediction  [chroma | mert]")
    for t in N:
        if t not in R: continue
        s = [r for r in R[t] if r["arm"] != "base" and r["eta"] >= 0.3]; ti = N.index(t)
        out = []
        for k in ("ch_pred", "mert_pred"):
            c = collections.Counter(r[k] for r in s); n = len(s)
            wrong = [(N[p], v / n) for p, v in c.most_common() if p != ti][:2]
            out.append(f"{c[ti]/n:.2f} (wrong: " + ", ".join(f"{a} {b:.2f}" for a, b in wrong) + ")")
        print(f"  {t:3s} {out[0]} | {out[1]}")
