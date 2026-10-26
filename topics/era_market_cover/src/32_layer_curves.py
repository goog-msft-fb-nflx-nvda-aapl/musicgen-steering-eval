"""Layer-wise probe curves (best CV accuracy on train per layer, C chosen per layer) for mix / vocals / accompaniment, MERT-v2 and MuQ, datasets A (decade) and B (market).
source: results/probe_{,muq_}{A,B}.json, key layer_cv_best_by_layer. output: results/figs/layer_curves.png. usage (Mac): python 32_layer_curves.py"""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = {"mix_norm": ("#0072B2", "mix"), "voc_norm": ("#D55E00", "vocals"), "acc_norm": ("#009E73", "accompaniment")}
fig, ax = plt.subplots(2, 2, figsize=(10, 6.2), sharey="row")
for i, ds in enumerate("AB"):
    for j, (enc, f) in enumerate((("MERT-v2-30s", f"results/probe_{ds}.json"), ("MuQ", f"results/probe_muq_{ds}.json"))):
        d = json.load(open(f))["conds"]; a = ax[i, j]
        for c, (col, name) in C.items():
            y = d[c]["layer_cv_best_by_layer"]; xs = sorted(map(int, y)); a.plot(xs, [y[str(x)] for x in xs], color=col, lw=2, label=name); a.plot(d[c]["layer"], d[c]["cv_acc"], "o", color=col, ms=7, mec="white", mew=1.5)
        a.set_title(f"{enc}, dataset {ds} ({'decade' if ds == 'A' else 'market'}, 6 classes)", fontsize=10); a.grid(alpha=.25); a.set_xlabel("layer") if i == 1 else None
        a.set_ylabel("5-fold CV accuracy on train") if j == 0 else None; a.axhline(1 / 6, color="gray", ls=":", lw=1)
        for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0, 0].legend(frameon=False, fontsize=9); fig.suptitle("Per-layer probe accuracy by stem (dot = layer chosen by CV; dotted = chance)", fontsize=11); fig.tight_layout(); fig.savefig("results/figs/layer_curves.png", dpi=150)
