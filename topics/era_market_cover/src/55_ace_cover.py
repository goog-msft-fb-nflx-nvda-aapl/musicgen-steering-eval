"""Third generator route: ACE-Step 1.5 (turbo DiT, MIT code; weights from the ACE-Step/Ace-Step1.5 Hugging Face repo, safetensors) in 'cover' mode on the ORIGINAL ACCOMPANIMENT STEM.
Same protocol as 07_generate.py: source = original accompaniment stem (first 30 s), caption = era prompt (same four prompts as 27_musecontrol.py), instrumental, task_type=cover, audio_cover_strength = --strength,
output (48 kHz) resampled to 24 kHz, trimmed, RMS-matched to the original accompaniment, remixed with the original vocal stem. Writes <outdir>/<id>_<era>_s<seed>_{acc,remix}.wav + picked.txt.
Run with ~/era_market/venv_ace from ~/era_market/ext/ACE-Step-1.5.  usage: python 55_ace_cover.py --strength 0.5 --outdir ace_gen_c05 --seeds 0 1 2"""
import argparse, os, sys, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from common import ROOT, SR
ERAS = {"neutral": "a song recording",
        "1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
ap = argparse.ArgumentParser(); ap.add_argument("--strength", type=float, default=0.5); ap.add_argument("--outdir", default="ace_gen_c05"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); open(os.path.join(out, "picked.txt"), "w").write(open(os.path.join(G, "picked.txt")).read()); tmp = os.path.join(ROOT, "ace_work"); os.makedirs(tmp, exist_ok=True)
dit = AceStepHandler(); llm = LLMHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda"), flush=True); n = 0
for s in ids:
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); src = os.path.join(tmp, s + "_acc.wav"); sf.write(src, acc0, SR)
    for era, prompt in ERAS.items():
        for sd in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{sd}")
            if os.path.exists(fp + "_remix.wav"): continue
            p = GenerationParams(task_type="cover", src_audio=src, caption=prompt, lyrics="[Instrumental]", instrumental=True, audio_cover_strength=a.strength, duration=30.0, inference_steps=8, shift=3.0, seed=sd, thinking=False)
            r = generate_music(dit, llm, p, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=tmp)
            if not r.success: print("FAIL", s, era, sd, r.error, flush=True); continue
            w = sf.read(r.audios[0]["path"], dtype="float32")[0]; w = w.mean(1) if w.ndim == 2 else w
            g = AF.resample(torch.from_numpy(np.ascontiguousarray(w)), 48000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, sd, n, flush=True); os.remove(r.audios[0]["path"])
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
