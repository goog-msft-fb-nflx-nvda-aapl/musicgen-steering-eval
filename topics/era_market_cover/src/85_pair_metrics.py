"""E07 paired metrics per clip with song-bootstrap CIs, from the av-benchmark caches written by run_avb_pair.sh (avbp/cache/{gt,pred}_<set>_<era>/).
KL-PaSST(clip) = KL(softmax(PaSST logits of the ORIGINAL clip) || softmax(logits of the generated remix)) as in av_bench/metrics/kl.py (mean over clips = the av-benchmark number; checked against avbp/out/<set>_<era>.json). Lower = the generated clip is classified more like the original.
CLAP-text(clip, text) = cosine(LAION-CLAP audio embedding of the generated remix [music_audioset_epoch_15_esc_90.14], text embedding of `text`).
Statistics per set and era e (1960s, 1980s, 2010s), per song = mean over its 3 seeds, then mean over the 24 songs:
  Delta KL   = KL(era e prompt) - KL(neutral prompt)                  (positive = era prompt moves the clip away from the original)
  Delta CLAP = cos(audio(era e), text(e)) - cos(audio(neutral), text(e)) (positive = era-prompted audio fits the era text better than neutral audio does; the same text on both sides)
95% CI: bootstrap over the 24 songs, 1000 resamples. Writes results/pair_<set>.json.
usage: python 85_pair_metrics.py set1 set2 ...   (run in ~/era_market, CPU)"""
import sys, os, json, numpy as np, torch
W = os.path.expanduser("~/era_market/avbp"); rng = np.random.default_rng(0); B = 1000
def ld(p): return torch.load(p, weights_only=True)
for s in sys.argv[1:]:
    per = {}; txt = {}
    for e in tuple(os.environ.get("EL", "1960s 1980s 2010s").split()) + ("neutral",):
        g = ld(f"{W}/cache/gt_{s}_{e}/passt_logits.pth"); p = ld(f"{W}/cache/pred_{s}_{e}/passt_logits.pth"); a = ld(f"{W}/cache/pred_{s}_{e}/clap_laion_audio.pth"); t = ld(f"{W}/cache/gt_{s}_{e}/clap_laion_text.pth")
        txt[e] = next(iter(t.values())).float(); kl = {}; au = {}
        for k in p:
            lg, lp = torch.log_softmax(g[k].float(), -1), torch.log_softmax(p[k].float(), -1); kl[k] = float((lg.exp() * (lg - lp)).sum()); au[k] = a[k].float()
        per[e] = (kl, au)
        j = f"{W}/out/{s}_{e}.json"
        if os.path.exists(j): print(s, e, "KL mean", round(np.mean(list(kl.values())), 4), "av-bench", round(json.load(open(j))["KL-PASST-softmax"], 4), "| CLAP own text", round(float(np.mean([torch.cosine_similarity(au[k], txt[e], dim=0) for k in au])), 4), "av-bench", round(json.load(open(j))["LAION-CLAP-Score"], 4))
    songs = sorted({k.split("_")[0] + "_" + k.split("_")[1] for k in per["neutral"][0]}); res = {}
    def song_mean(e, fn): return np.array([np.mean([fn(k) for k in per[e][0] if k.startswith(x + "_")]) for x in songs])
    for e in tuple(os.environ.get("EL", "1960s 1980s 2010s").split()):
        klE, klN = song_mean(e, lambda k: per[e][0][k]), song_mean("neutral", lambda k: per["neutral"][0][k])
        cE = song_mean(e, lambda k: float(torch.cosine_similarity(per[e][1][k], txt[e], dim=0))); cN = song_mean("neutral", lambda k: float(torch.cosine_similarity(per["neutral"][1][k], txt[e], dim=0)))
        r = {}
        for name, d in (("delta_kl", klE - klN), ("delta_clap", cE - cN)):
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]; r[name] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        r.update(kl_era=float(klE.mean()), kl_neutral=float(klN.mean()), clap_era=float(cE.mean()), clap_neutral=float(cN.mean())); res[e] = r
        print(s, e, {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in r.items()})
    json.dump(res, open(os.path.expanduser(f"~/era_market/results/pair_{s}.json"), "w"), indent=1)
