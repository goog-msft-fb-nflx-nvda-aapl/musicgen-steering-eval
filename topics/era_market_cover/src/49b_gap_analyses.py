"""Gap analyses for the TA description (CPU, cached features). usage: 49b_gap_analyses.py A|B > results/gap_<ds>.txt ; writes results/gap_<ds>.json and results/gap_<ds>_clips.csv
Pooled labeled clips (train+val; the manifest has no artist ids, so folds are not artist-grouped), 5-fold stratified out-of-fold (OOF) predictions with the per-stem layer chosen on TRAIN (probe_<ds>.json), averaged over 3 fold seeds.
(1) song-level hardness: clips wrong under all three of mix/vocals/accompaniment, ordinal error; (2) what hard clips share: vocal share, Whisper language, loudness, Discogs style tags; (3) genre/style proxy: tag-only and EffNet-only probes,
MERT+tags vs MERT, accuracy inside style clusters; (4) market: language-vs-market mismatch (modal language per market from TRAIN)."""
import sys, json, csv, os, numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from common import ROOT, LABELS, manifest
ds = sys.argv[1]; E = lambda n: np.load(os.path.join(ROOT, "emb", f"{ds}_{n}.npy"), allow_pickle=True)
rows = manifest(ds); lab = LABELS[ds]; y_all = np.array([lab.index(r["label"]) if r["label"] else -1 for r in rows]); idx = np.where(y_all >= 0)[0]; y = y_all[idx]
split = np.array([r["split"] for r in rows])[idx]; P = json.load(open(os.path.join(ROOT, "results", f"probe_{ds}.json")))["conds"]
def oof(X, seeds=(0, 1, 2), C=0.01):
    pr = np.zeros((len(y), len(lab)))
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr]); pr[te] += m.predict_proba(X[te])
    return pr / len(seeds)
acc = lambda p, m=None: float((p.argmax(1) == y)[m if m is not None else slice(None)].mean())
feats = {c: E(c)[idx][:, P[c]["layer"]].astype(np.float32) for c in ("mix_norm", "voc_norm", "acc_norm")}
PR = {c: oof(X) for c, X in feats.items()}; out = {"n": len(y), "oof_acc": {c: acc(p) for c, p in PR.items()}}
print(f"== dataset {ds}: n={len(y)} pooled train+val, 5-fold OOF x3 seeds (not artist-grouped)\nOOF accuracy:", {c: round(v, 3) for c, v in out["oof_acc"].items()})
wrong = np.stack([PR[c].argmax(1) != y for c in PR], 1); nw = wrong.sum(1)
print("clips wrong under 0/1/2/3 of the three stems:", [int((nw == k).sum()) for k in range(4)], "; expected number wrong in all three if stem errors were independent:", round(len(y) * float(np.prod([1 - v for v in out["oof_acc"].values()])), 1))
if ds == "A":
    for c, p in PR.items(): e = np.abs(p.argmax(1) - y); out.setdefault("ordinal", {})[c] = {"mean_abs_decade_err": float(e.mean()), "within1": float((e <= 1).mean())}
    print("ordinal:", {c: {k: round(v, 3) for k, v in d.items()} for c, d in out["ordinal"].items()})
# per-class OOF confusion of the mix probe
cm = np.zeros((len(lab), len(lab)), int)
for t, p in zip(y, PR["mix_norm"].argmax(1)): cm[t, p] += 1
print("mix OOF confusion (rows true, cols pred), order", lab); print(cm)
out["confusion_mix"] = cm.tolist()
# agreement of hardness across the three stems
hard = nw == 3; easy = nw == 0
# covariates
voc_share = E("vocshare")[idx]; lid_v = E("lid_voc")[idx]; langs = json.load(open(os.path.join(ROOT, "emb", "lid_langs.json"))); toplang = np.array([langs[i] for i in lid_v.argmax(1)])
stats = E("stats_mix")[idx]; loud = stats[:, 0]
def cmp(name, v):
    a, b = v[hard], v[easy]; p = mannwhitneyu(a, b).pvalue if len(a) > 3 and len(b) > 3 else float("nan")
    print(f"  {name}: hard median {np.median(a):.3f} (n={len(a)}) vs easy median {np.median(b):.3f} (n={len(b)}), Mann-Whitney p={p:.3g}"); out.setdefault("hard_vs_easy", {})[name] = {"hard": float(np.median(a)), "easy": float(np.median(b)), "p": float(p)}
print("\n== hard (wrong in all 3 stems) vs easy (right in all 3)"); cmp("vocal share", voc_share); cmp("integrated loudness-like stat (stats col 0)", loud)
lc = {"hard": Counter(toplang[hard]).most_common(6), "easy": Counter(toplang[easy]).most_common(6), "all": Counter(toplang).most_common(6)}; print("  top vocal-language (Whisper LID argmax):", lc); out["lang_counts"] = lc
for L in [l for l, _ in lc["all"][:4]]:
    m = toplang == L; print(f"  OOF acc mix/voc/acc where language={L} (n={int(m.sum())}):", [round(acc(PR[c], m), 3) for c in PR])
# genre / style proxy
tags = E("dgtags")[idx]; dge = E("dgemb")[idx]; classes = json.load(open(os.path.join(ROOT, "emb", "dg_classes.json")))
print("\n== genre/style proxy (Discogs-EffNet on the original mix)")
Pt = oof(tags); Pe = oof(dge); Pmt = oof(np.hstack([feats["mix_norm"], tags]), C=0.01)
out["tags_only_acc"] = acc(Pt); out["effnet_only_acc"] = acc(Pe); out["mert_plus_tags_acc"] = acc(Pmt)
print(f"  OOF acc: style-tags only {acc(Pt):.3f}; EffNet embedding only {acc(Pe):.3f}; MERT mix {acc(PR['mix_norm']):.3f}; MERT mix + tags {acc(Pmt):.3f} (chance {1/len(lab):.3f})")
b = (Pmt.argmax(1) == y); a0 = (PR["mix_norm"].argmax(1) == y); print(f"  MERT+tags vs MERT: only-concat-right {int((b & ~a0).sum())}, only-MERT-right {int((~b & a0).sum())}")
tr_mask = split == "train"; top_style = {}
for k, l in enumerate(lab):
    mt = tags[y == k].mean(0) - tags.mean(0); top_style[l] = [(classes[i], round(float(mt[i]), 3)) for i in np.argsort(-mt)[:5]]
    print(f"  most over-represented styles in {l}: {top_style[l]}")
out["top_style"] = top_style
km = KMeans(6, n_init=10, random_state=0).fit(dge); cl = km.labels_; print("  accuracy of MERT mix probe inside 6 EffNet style clusters (n, majority-class share, MERT acc, tags-only acc):")
for k in range(6):
    m = cl == k; maj = np.bincount(y[m], minlength=len(lab)).max() / m.sum(); print(f"    cluster {k}: n={int(m.sum())} majority share {maj:.2f}, MERT {acc(PR['mix_norm'], m):.3f}, tags-only {acc(Pt, m):.3f}")
    out.setdefault("clusters", []).append({"n": int(m.sum()), "majority": float(maj), "mert": acc(PR["mix_norm"], m), "tags": acc(Pt, m)})
# hard clips: which styles
mt = tags[hard].mean(0) - tags.mean(0); print("  styles over-represented in HARD clips:", [(classes[i], round(float(mt[i]), 3)) for i in np.argsort(-mt)[:6]]); out["hard_styles"] = [(classes[i], float(mt[i])) for i in np.argsort(-mt)[:6]]
# market: language mismatch
if ds == "B":
    print("\n== market vs language (Whisper LID argmax on the vocal stem is a proxy for sung language; native language per market is an assumption fixed by us, not read from the data)")
    native = {"US": "en", "UK": "en", "Brazil": "pt", "Spain": "es", "Germany": "de", "Italy": "it"}
    comp = {l: Counter(toplang[y == k]).most_common(4) for k, l in enumerate(lab)}
    for l in lab: print(f"  {l}: top languages {comp[l]}")
    nat = np.array([toplang[i] == native[lab[y[i]]] for i in range(len(y))]); eng = np.array([toplang[i] == "en" and native[lab[y[i]]] != "en" for i in range(len(y))])
    oth = ~nat & ~eng; grp = {"native language": nat, "English in a non-English market (ES/DE/IT)": eng, "other detected language": oth}
    out["lang_groups"] = {}
    for g, m in grp.items():
        print(f"  {g}: n={int(m.sum())}; OOF acc mix/voc/acc = {[round(acc(PR[c], m), 3) for c in PR]}"); out["lang_groups"][g] = {"n": int(m.sum()), "acc": {c: acc(PR[c], m) for c in PR}}
    for l in ("Spain", "Germany", "Italy"):
        k = lab.index(l); me = (y == k) & eng; mn = (y == k) & nat
        print(f"  {l}: English-sung n={int(me.sum())} mix acc {acc(PR['mix_norm'], me):.3f}; native-language n={int(mn.sum())} mix acc {acc(PR['mix_norm'], mn):.3f}")
    errs = cm.sum() - np.trace(cm); us_uk = cm[0, 1] + cm[1, 0]; print(f"  mix-probe errors total {int(errs)}; US<->UK confusions {int(us_uk)}; errors whose true and predicted markets are both in {{US,UK,Germany,Italy,Spain}} (English dominates vocals): see confusion matrix")
    out["native_assumption"] = native
json.dump(out, open(os.path.join(ROOT, "results", f"gap_{ds}.json"), "w"))
with open(os.path.join(ROOT, "results", f"gap_{ds}_clips.csv"), "w") as f:
    w = csv.writer(f); w.writerow(["sample_id", "true", "pred_mix", "pred_voc", "pred_acc", "n_wrong", "top_vocal_language", "vocal_share"])
    for k, i in enumerate(idx): w.writerow([rows[i]["sample_id"], lab[y[k]], *[lab[PR[c][k].argmax()] for c in PR], int(nw[k]), toplang[k], round(float(voc_share[k]), 3)])
