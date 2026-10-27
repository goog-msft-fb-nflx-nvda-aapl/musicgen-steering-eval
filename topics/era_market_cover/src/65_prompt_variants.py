"""Prompt elaboration ablation (G05): same generator, protocol and scoring as 07_generate.py (MusicGen-Melody g=3, chroma of the original mix, original-vocal remix), different text for the target eras:
word  = "the 1980s" (decade only); short = "a 1980s song"; tags = comma-separated keywords. The detailed one-sentence prompts of 07_generate.py are the existing gen/ runs. Neutral rows are copied from gen/.
usage: python 65_prompt_variants.py --variant tags --outdir gen_pv_tags --seeds 0 1 2"""
import argparse, os, shutil, glob, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest
V = {"word": {"1960s": "the 1960s", "1980s": "the 1980s", "2010s": "the 2010s"},
     "short": {"1960s": "a 1960s song", "1980s": "a 1980s song", "2010s": "a 2010s song"},
     "tags": {"1960s": "1960s, pop, soul, mono tape, live drums, upright bass, brass, organ, analog warmth", "1980s": "1980s, synth pop, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy", "2010s": "2010s, modern pop, sub bass, punchy compressed drums, wide polished synths, loud mastering"}}
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True, choices=list(V)); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--guidance", type=float, default=3.0); a = ap.parse_args()
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(G, "picked.txt"), os.path.join(out, "picked.txt"))
for f in glob.glob(os.path.join(G, "*_neutral_s*_*.wav")):
    if int(os.path.basename(f).split("_s")[1][0]) in a.seeds: shutil.copy(f, out)
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval(); n = 0
for s in ids:
    mix = sf.read(rows[s]["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era, prompt in V[a.variant].items():
        for seed in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            inp = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt").to("cuda"); inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
            torch.manual_seed(seed)
            with torch.no_grad(): o = model.generate(**inp, do_sample=True, guidance_scale=a.guidance, max_new_tokens=1500)
            g = AF.resample(o[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, seed, n, flush=True)
print("done", n)
