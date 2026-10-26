"""Learnable layer-weighted-sum probe (MERT-paper style protocol) as a robustness check of the per-layer CV selection.
per condition: softmax-weighted sum over all layers of the standardized frozen embeddings (weights learned jointly with a linear head, Adam, weight decay, full batch), epochs and weight decay chosen by 5-fold CV on train,
scored once on validation; compared with the best-single-layer probe of 04_probe.py. usage: python 25_lws_probe.py --ds A --prefix "" --out results/lws_mert_A.json"""
import argparse, json, os, numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from common import LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--prefix", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--emb", default=os.path.expanduser("~/era_market/emb")); ap.add_argument("--conds", nargs="+", default=["mix_norm", "voc_norm", "acc_norm", "recomb_norm"]); a = ap.parse_args()
rows = manifest(a.ds); K = len(LABELS[a.ds]); split = np.array([r["split"] for r in rows]); y = np.array([LABELS[a.ds].index(r["label"]) if r["label"] else -1 for r in rows])
tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]; dev = "cuda" if torch.cuda.is_available() else "cpu"
def train(X, yy, wd, epochs, seed=0):
    torch.manual_seed(seed); L, D = X.shape[1], X.shape[2]; w = torch.zeros(L, device=dev, requires_grad=True); lin = torch.nn.Linear(D, K).to(dev)
    opt = torch.optim.AdamW([{"params": [w], "weight_decay": 0.0}, {"params": lin.parameters(), "weight_decay": wd}], lr=3e-3); yt = torch.from_numpy(yy).to(dev)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(lin((torch.softmax(w, 0)[None, :, None] * X).sum(1)), yt); loss.backward(); opt.step()
    return w.detach(), lin
def predict(w, lin, X): 
    with torch.no_grad(): return lin((torch.softmax(w, 0)[None, :, None] * X).sum(1)).argmax(1).cpu().numpy()
single = json.load(open(os.path.join(os.path.expanduser("~/era_market/results"), f"probe_{a.prefix}{a.ds}.json")))["conds"]
res = {}; rng = np.random.default_rng(0); boots = rng.integers(0, len(va), size=(2000, len(va)))
for c in a.conds:
    E = np.load(os.path.join(a.emb, f"{a.ds}_{a.prefix}{c}.npy")).astype(np.float32); mu, sd = E[tr].mean(0, keepdims=True), E[tr].std(0, keepdims=True) + 1e-6
    X = torch.from_numpy((E - mu) / sd).to(dev); best = (-1, None)
    for wd in (1e-2, 1e-1, 1.0):
        for ep in (100, 300):
            accs = []
            for f_tr, f_va in StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]):
                w, lin = train(X[tr[f_tr]], y[tr[f_tr]], wd, ep); accs.append((predict(w, lin, X[tr[f_va]]) == y[tr[f_va]]).mean())
            if np.mean(accs) > best[0]: best = (float(np.mean(accs)), (wd, ep))
    wd, ep = best[1]; w, lin = train(X[tr], y[tr], wd, ep); ok = (predict(w, lin, X[va]) == y[va]).astype(float); sw = torch.softmax(w, 0).cpu().numpy()
    res[c] = {"cv_acc": best[0], "wd": wd, "epochs": ep, "val_acc": float(ok.mean()), "val_ci95": [float(np.quantile(ok[boots].mean(1), .025)), float(np.quantile(ok[boots].mean(1), .975))],
              "single_layer_val_acc": single[c]["val_acc"], "single_layer": single[c]["layer"], "layer_weights": sw.round(4).tolist(), "top_layers": np.argsort(-sw)[:3].tolist()}
    print(a.ds, a.prefix or "mert_", c, f"LWS val {ok.mean():.3f} (cv {best[0]:.3f}, wd {wd}, ep {ep}) | single-layer L{single[c]['layer']} val {single[c]['val_acc']:.3f} | top layers {res[c]['top_layers']}", flush=True)
json.dump(res, open(a.out, "w"))
