"""One row per generation set: Audiobox Aesthetics (PQ, CE; mean over the 24 songs of the song-averaged score, era prompts pooled vs neutral) and KAD-style margin delta (MERT, per decade; delta = margin(era prompt) - margin(neutral), as in results/kad_*.json). Output results/quality_kad_table.txt. usage: python src/81_quality_kad_table.py"""
import glob, json, os, numpy as np
rows = []
for qf in sorted(glob.glob("results/quality_*.json")):
    name = os.path.basename(qf)[8:-5]; q = json.load(open(qf)); songs = sorted({n.split("|")[0] for n in q if not n.startswith("noise")})
    def m(cond, k): return float(np.mean([np.mean([v[k] for n, v in q.items() if n.startswith(s + "|" + c + "|")]) for s in songs for c in cond]))
    kf = f"results/kad_{name}.json"; kd = json.load(open(kf)) if os.path.exists(kf) else None
    kad = " / ".join(f"{kd['mert'][d]['delta']:+.3f}" for d in ("1960s", "1980s", "2010s")) if kd else "-"
    rows.append((name, m(["neutral"], "PQ"), m(["1960s", "1980s", "2010s"], "PQ"), m(["neutral"], "CE"), m(["1960s", "1980s", "2010s"], "CE"), kad))
L = ["set | PQ neutral | PQ era | CE neutral | CE era | KAD delta MERT 1960s / 1980s / 2010s"] + [f"{n} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {k}" for n, a, b, c, d, k in rows]
open("results/quality_kad_table.txt", "w").write("\n".join(L)); print("\n".join(L))
