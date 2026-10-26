"""Analysis-half metrics on stored embeddings (CPU). For each dataset / encoder / stem condition (mix_norm, voc_norm, acc_norm, recomb_norm), with the layer and C that 04_probe.py chose by train CV:
validation accuracy, log-loss (nats), macro-F1, and for dataset A the ordinal metrics QWK and MAE in decades (argmax and expected-decade), all with a bootstrap 95% CI over validation clips;
online-code MDL (Voita & Titov: blocks of 1,2,4,..,~train size; first block coded uniformly) in bits per training clip; control-task selectivity (Hewitt & Liang style: random labels with the real label frequencies, fixed per clip,
probe trained on them; selectivity = real val acc - control val acc). usage: python 22_analysis_metrics.py --out results/analysis_metrics.json > results/analysis_metrics.txt"""
import argparse, json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score, f1_score, log_loss
from common import LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--emb", default=os.path.expanduser("~/era_market/emb")); ap.add_argument("--conds", nargs="+", default=["mix_norm", "voc_norm", "acc_norm", "recomb_norm"]); a = ap.parse_args()
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results") if not os.path.exists(os.path.expanduser("~/era_market/results")) else os.path.expanduser("~/era_market/results")
def fit(E, y, C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(E, y)
def full_proba(m, X, K):
    p = np.zeros((len(X), K)); p[:, m.classes_] = m.predict_proba(X); return p
out = {}; rng = np.random.default_rng(0)
for ds in ("A", "B"):
    rows = manifest(ds); K = len(LABELS[ds]); split = np.array([r["split"] for r in rows]); y = np.array([LABELS[ds].index(r["label"]) if r["label"] else -1 for r in rows])
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]; boots = rng.integers(0, len(va), size=(1000, len(va)))
    for enc, pre, pj in (("mert", "", f"probe_{ds}.json"), ("muq", "muq_", f"probe_muq_{ds}.json")):
        cfg = json.load(open(os.path.join(R, pj)))["conds"]
        for c in a.conds:
            if c not in cfg: continue
            l, C = cfg[c]["layer"], cfg[c]["C"]; E = np.load(os.path.join(a.emb, f"{ds}_{pre}{c}.npy")).astype(np.float32)[:, l]
            m = fit(E[tr], y[tr], C); P = full_proba(m, E[va], K); yv = y[va]; pred = P.argmax(1); ok = (pred == yv).astype(float)
            met = {"layer": int(l), "C": C, "acc": float(ok.mean()), "logloss": float(log_loss(yv, np.clip(P, 1e-9, 1), labels=range(K))), "macro_f1": float(f1_score(yv, pred, average="macro"))}
            bs = {"acc": [], "logloss": []}
            nll = -np.log(np.clip(P[np.arange(len(yv)), yv], 1e-9, 1))
            ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            met["acc_ci"] = ci(ok[boots].mean(1)); met["logloss_ci"] = ci(nll[boots].mean(1))
            if ds == "A":
                ed = P @ np.arange(K); met["qwk"] = float(cohen_kappa_score(yv, pred, weights="quadratic")); met["mae_decades_argmax"] = float(np.abs(pred - yv).mean()); met["mae_decades_expected"] = float(np.abs(ed - yv).mean())
                met["mae_decades_expected_ci"] = ci(np.abs(ed - yv)[boots].mean(1)); met["mae_majority_baseline"] = float(np.abs(np.bincount(y[tr]).argmax() - yv).mean())
                met["qwk_ci"] = ci([cohen_kappa_score(yv[b], pred[b], weights="quadratic") for b in boots[:300]])
            # online-code MDL on the training set (random order, fixed seed)
            order = np.random.default_rng(1).permutation(tr); n = len(order); cuts = [0]
            while cuts[-1] < n: cuts.append(min(n, max(1, cuts[-1] * 2)))
            cuts = sorted(set(cuts + [n])); L = cuts[1] * np.log2(K)
            for i in range(1, len(cuts) - 1):
                idx, nxt = order[:cuts[i]], order[cuts[i]:cuts[i + 1]]
                if len(set(y[idx])) < 2: L += len(nxt) * np.log2(K); continue
                mm = fit(E[idx], y[idx], C); L += float(-np.log2(np.clip(full_proba(mm, E[nxt], K)[np.arange(len(nxt)), y[nxt]], 1e-9, 1)).sum())
            met["mdl_bits_per_clip"] = float(L / n); met["uniform_bits_per_clip"] = float(np.log2(K))
            # control task
            sel = []
            for seed in range(3):
                g = np.random.default_rng(100 + seed); yc = y.copy(); lab = y >= 0; yc[lab] = g.choice(K, size=lab.sum(), p=np.bincount(y[tr], minlength=K) / len(tr))
                mc = fit(E[tr], yc[tr], C); sel.append(float(ok.mean() - (mc.predict(E[va]) == yc[va]).mean()))
            met["selectivity"] = float(np.mean(sel)); met["control_acc"] = float(ok.mean() - np.mean(sel))
            out[f"{ds}/{enc}/{c}"] = met
            print(f"{ds} {enc:4s} {c:11s} L{l:2d} acc {met['acc']:.3f} {met['acc_ci'][0]:.2f}-{met['acc_ci'][1]:.2f} | logloss {met['logloss']:.3f} | f1 {met['macro_f1']:.3f} | " +
                  (f"QWK {met['qwk']:.3f} MAEexp {met['mae_decades_expected']:.2f} (majority {met['mae_majority_baseline']:.2f}) | " if ds == "A" else "") +
                  f"MDL {met['mdl_bits_per_clip']:.2f}/{met['uniform_bits_per_clip']:.2f} bits | control acc {met['control_acc']:.3f} selectivity {met['selectivity']:+.3f}", flush=True)
json.dump(out, open(a.out, "w"), indent=1)
