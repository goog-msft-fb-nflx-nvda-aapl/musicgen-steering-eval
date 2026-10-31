"""Six-decade controllability table: for each decade prompt (1960s ... 2010s) the per-song change of P(target decade | era prompt) - P(. | neutral) and the mean expected-decade index of the generated accompaniment, from the codec-matched MERT / MuQ probes.
Main-decade eval json (1960s/1980s/2010s) and extra-decade eval json (1970s/1990s/2000s) of the same configuration are joined; 24 songs, seeds averaged, 95% song-bootstrap CI.
usage: python 78_six_decades.py label=main_eval.json+extra_eval.json ... -> results/six_decades.{json,txt}"""
import sys, json, numpy as np
D = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]; rng = np.random.default_rng(0); out = {}; txt = []
for arg in sys.argv[1:]:
    label, rest = arg.split("=", 1); R = []
    for f in rest.split("+"): R += json.load(open(f))["rows"]
    songs = sorted({r["clip"] for r in R}); line = [f"{label} (n_songs={len(songs)})"]; out[label] = {}
    for enc in ("mert", "muq"):
        cells = []
        for t, e in enumerate(D):
            Pm = lambda c: np.array([np.mean([r[f"{enc}_probs"][t] for r in R if r["clip"] == s and r["cond"] == c and r["seed"] in (0, 1, 2)]) for s in songs])
            d = Pm(e) - Pm("neutral"); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            ed = np.mean([np.sum(np.array(r[f"{enc}_probs"]) * np.arange(6)) for r in R if r["cond"] == e]); top1 = np.mean([int(np.argmax(r[f"{enc}_probs"]) == t) for r in R if r["cond"] == e])
            out[label][f"{enc}_{e}"] = {"delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "expected_decade": float(ed), "top1_target": float(top1), "pos": int((d > 0).sum())}
            cells.append(f"{e} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] E={ed:.2f} top1={top1:.2f}")
        line.append(f"  {enc}: " + " | ".join(cells))
    txt += line; print("\n".join(line))
json.dump(out, open("results/six_decades.json", "w"), indent=1); open("results/six_decades.txt", "w").write("\n".join(txt) + "\n")
