"""G06 step 2/3: generate with a given prompt set, same protocol as 07_generate.py (MusicGen-Melody g=3, chroma of the original mix, original-vocal remix).
--split train: 24 TRAIN songs (4 per decade, manifest order), seed 0, used to SELECT among LLM candidates (no validation song is touched during selection).
--split validation: the 24 validation songs of gen/ (picked.txt), seeds 0 1 2, used to TEST the selected prompts. Neutral rows: validation copied from gen/, train generated here.
prompts: --prompts <json from 69_llm_prompts.py> --cand k (candidate index) or --select <selection json: {decade: prompt}>.
usage: python 70_prompt_set_gen.py --split train --prompts results/llm_prompts_x.json --cand 3 --outdir gen_g06_x_c3"""
import argparse, os, json, shutil, glob, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest, LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["train", "validation"]); ap.add_argument("--prompts"); ap.add_argument("--cand", type=int); ap.add_argument("--select"); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--guidance", type=float, default=3.0); a = ap.parse_args()
if a.select: P = json.load(open(a.select))
else: P = {d: v[a.cand] for d, v in json.load(open(a.prompts))["prompts"].items()}
P = dict(P); rows = {r["sample_id"]: r for r in manifest("A")}; out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True)
if a.split == "train":
    tr = [r for r in manifest("A") if r["split"] == "train"]; ids = []
    for lab in LABELS["A"]: ids += [r["sample_id"] for r in tr if r["label"] == lab][:4]
    lab_of = {r["sample_id"]: r["label"] for r in tr}; ERAS = {"neutral": "a song recording", **P}
else:
    G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; lab_of = {l.split(",")[0]: l.split(",")[1] for l in open(os.path.join(G, "picked.txt")).read().split("\n")}; ERAS = dict(P)
    for f in glob.glob(os.path.join(G, "*_neutral_s*_*.wav")):
        if int(os.path.basename(f).split("_s")[1][0]) in a.seeds: shutil.copy(f, out)
open(os.path.join(out, "picked.txt"), "w").write("\n".join(f"{s},{lab_of[s]}" for s in ids)); json.dump(ERAS, open(os.path.join(out, "prompts.json"), "w"), indent=1)
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval(); n = 0
for s in ids:
    mix = sf.read(rows[s]["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era, prompt in ERAS.items():
        for seed in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            inp = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt").to("cuda"); inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
            torch.manual_seed(seed)
            with torch.no_grad(): o = model.generate(**inp, do_sample=True, guidance_scale=a.guidance, max_new_tokens=1500)
            g = AF.resample(o[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, seed, n, flush=True)
print("done", n)
