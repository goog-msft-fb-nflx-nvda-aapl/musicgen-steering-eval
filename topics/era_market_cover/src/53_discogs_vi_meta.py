"""Label-semantics statistics on the Discogs-VI metadata (Araz et al. 2024; metadata CC BY-NC-SA 4.0), streamed from main.zip, no audio. usage: unzip -p main.zip main/Discogs-VI-20240701.jsonl | python 53_discogs_vi_meta.py > results/discogs_vi_meta.txt
Per version we use the FIRST track's metadata (country, released year, first track artist id). Artist origin is NOT in the data (the artist file has only names, aliases, members), so "artist origin" is approximated by the artist's most frequent
release country over all their versions (artists with >= 5 versions) -- a proxy, not origin.
Outputs: field coverage, country/decade counts, per-clique multiplicity of countries and decades (cliques with >= 2 versions), cross-market ambiguity for the six dataset-B markets in the 1980s, and release-country vs artist-modal-country divergence."""
import sys, json, re, numpy as np
from collections import Counter, defaultdict
MK = {"US": "US", "UK": "UK", "Brazil": "Brazil", "Spain": "Spain", "Germany": "Germany", "Italy": "Italy"}
rec = []   # (clique, artist, country, year)
n_v = 0; miss_country = 0; miss_year = 0
for ci, line in enumerate(sys.stdin):
    c = json.loads(line)
    for v in c["versions"]:
        n_v += 1; t = v["tracks"][0] if v["tracks"] else {}
        ct = (t.get("country") or "").strip(); yr = (t.get("released") or "")[:4]; yr = int(yr) if re.fullmatch(r"\d{4}", yr) else 0
        a = t["track_artist_ids"][0] if t.get("track_artist_ids") else ""
        miss_country += (ct == ""); miss_year += (yr == 0); rec.append((ci, a, ct, yr))
    if ci % 100000 == 0: print("clique", ci, file=sys.stderr, flush=True)
print(f"versions {n_v}; cliques {rec[-1][0] + 1}; missing country {miss_country} ({miss_country / n_v:.3f}); missing/invalid year {miss_year} ({miss_year / n_v:.3f})")
cc = Counter(r[2] for r in rec if r[2]); print("top countries:", cc.most_common(15)); print("six dataset-B markets (Discogs names US, UK, Brazil, Spain, Germany, Italy):", {m: cc.get(m, 0) for m in MK})
dec = Counter((r[3] // 10) * 10 for r in rec if r[3] >= 1950); print("versions per release decade:", sorted(dec.items()))
byc = defaultdict(list)
for r in rec: byc[r[0]].append(r)
multi = [v for v in byc.values() if len(v) >= 2]; print(f"cliques with >=2 versions: {len(multi)}")
nc = np.array([len({r[2] for r in v if r[2]}) for v in multi]); nd = np.array([len({(r[3] // 10) * 10 for r in v if r[3] >= 1950}) for v in multi])
print(f"distinct release countries per clique: mean {nc.mean():.2f}, share with >=2 {np.mean(nc >= 2):.3f}, >=3 {np.mean(nc >= 3):.3f}; distinct release decades per clique: mean {nd.mean():.2f}, share with >=2 {np.mean(nd >= 2):.3f}, >=3 {np.mean(nd >= 3):.3f}")
# cross-market ambiguity: versions released in the 1980s in one of the six markets; does their clique contain a 1980s version in another of the six markets?
n80 = 0; both = 0; pair = Counter(); other_dec = 0
for v in multi:
    m80 = [r for r in v if 1980 <= r[3] <= 1989 and r[2] in MK]
    for r in m80:
        n80 += 1; others = {x[2] for x in m80 if x[2] != r[2]}; both += bool(others); other_dec += any(x[3] and (x[3] // 10) != 198 for x in v)
        for o in others: pair[(r[2], o)] += 1
print(f"1980s versions in the six markets inside cliques with >=2 versions: {n80}; share whose clique has a 1980s version in a DIFFERENT one of the six markets: {both / max(n80, 1):.3f}; share whose clique has a version outside the 1980s: {other_dec / max(n80, 1):.3f}")
print("most frequent (market, other market) pairs among those:", pair.most_common(10))
# artist-modal-country divergence
per = defaultdict(Counter)
for r in rec:
    if r[1] and r[2]: per[r[1]][r[2]] += 1
modal = {a: c.most_common(1)[0][0] for a, c in per.items() if sum(c.values()) >= 5}
def div(rs): rs = [r for r in rs if r[1] in modal and r[2]]; return (sum(r[2] != modal[r[1]] for r in rs) / len(rs), len(rs)) if rs else (float("nan"), 0)
d, n = div(rec); print(f"release country != artist's most frequent release country (artists with >=5 versions): {d:.3f} over {n} versions")
for m in MK:
    d, n = div([r for r in rec if r[2] == m]); d80, n80_ = div([r for r in rec if r[2] == m and 1980 <= r[3] <= 1989]); print(f"  releases in {m}: {d:.3f} (n={n}); 1980s only: {d80:.3f} (n={n80_})")
