"""Sampling-parameter ablation (G07). Same protocol as 07_generate.py (MusicGen-Melody, guidance 3, chroma of the original mix, original-vocal remix), but with explicit sampling settings. Baseline of the main runs = library defaults of
transformers 5.x for do_sample=True: temperature 1.0, top_k 50 (generation/configuration_utils.py default dict), top_p 1.0. Facebook's reference MusicGen sampling uses top_k 250.
configs: k250 (top_k 250), k0p09 (top_k 0, top_p 0.9), t08 (temperature 0.8, top_k 50), t12 (temperature 1.2, top_k 50). All four eras are generated per config (neutral included, for the paired delta).
usage: python 68_sampling_params.py --config k250 --outdir gen_sp_k250 --seeds 0 1"""
import argparse, os, shutil, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest
ERAS = {"neutral": "a song recording",
        "1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
CFG = {"k250": dict(top_k=250), "k0p09": dict(top_k=0, top_p=0.9), "t08": dict(temperature=0.8, top_k=50), "t12": dict(temperature=1.2, top_k=50), "t12k250": dict(temperature=1.2, top_k=250), "t14k250": dict(temperature=1.4, top_k=250)}   # last two: entropy-matching candidates (src/72_entropy_match.py)
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True, choices=list(CFG)); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); a = ap.parse_args()
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(G, "picked.txt"), os.path.join(out, "picked.txt"))
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval(); n = 0
for s in ids:
    mix = sf.read(rows[s]["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era, prompt in ERAS.items():
        for seed in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            inp = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt").to("cuda"); inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
            torch.manual_seed(seed)
            with torch.no_grad(): o = model.generate(**inp, do_sample=True, guidance_scale=3.0, max_new_tokens=1500, **CFG[a.config])
            g = AF.resample(o[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, seed, n, flush=True)
print("done", n)
