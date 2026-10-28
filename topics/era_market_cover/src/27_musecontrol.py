"""Second generator: MuseControlLite (Stable Audio Open 1.0 + melody_stereo adapter, ICML 2025). Same task as 07_generate.py: melody condition extracted from the ORIGINAL MIX + era text prompt (same ERAS prompts) -> accompaniment-like audio;
first 30 s resampled to 24 kHz mono, RMS-matched to the original accompaniment stem, remix with the original vocal stem. Writes mcl_gen/<id>_<era>_s<seed>_{acc,remix}.wav + picked.txt (same songs as gen/).
run from ~/era_market/ext/MuseControlLite with venv_mcl:  python ../../src/27_musecontrol.py --seeds 0 1"""
import argparse, os, sys, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from MuseControlLite_setup import setup_MuseControlLite, initialize_condition_extractors, process_musical_conditions
from config_inference import get_config
from common import ROOT, SR, manifest
ERAS = {"neutral": "a song recording",
        "1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--limit_gen", type=int, default=0); ap.add_argument("--outdir", default="mcl_gen"); ap.add_argument("--gtext", type=float, default=None); ap.add_argument("--gcon", type=float, default=None); ap.add_argument("--gaudio", type=float, default=None); a = ap.parse_args()
cfg = get_config(); cfg["show_result_and_plt"] = False
for k, v in (("guidance_scale_text", a.gtext), ("guidance_scale_con", a.gcon), ("guidance_scale_audio", a.gaudio)):
    if v is not None: cfg[k] = v   # G08b sweep; defaults of the repo config (7.0, 1.5, 1.0) when not given
cfg["condition_type"] = ["melody_stereo"]; cfg["output_dir"] = os.path.join(ROOT, "mcl_work"); os.makedirs(cfg["output_dir"], exist_ok=True)
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); open(os.path.join(out, "picked.txt"), "w").write(open(os.path.join(G, "picked.txt")).read())
ext, tck = initialize_condition_extractors(cfg); model = setup_MuseControlLite(cfg, torch.float16, tck).to("cuda"); n = 0
for s in ids:
    mx = AF.resample(torch.from_numpy(sf.read(rows[s]["path"], dtype="float32")[0]), SR, 44100).numpy(); cond_wav = os.path.join(cfg["output_dir"], s + "_mix44k.wav"); sf.write(cond_wav, np.stack([mx, mx], 1), 44100)   # melody extractor needs a 44.1 kHz file
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    for era, prompt in ERAS.items():
        for sd in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{sd}")
            if os.path.exists(fp + "_remix.wav"): continue
            with torch.no_grad():
                fc, fca = process_musical_conditions(cfg, cond_wav, ext, cfg["output_dir"], 0, torch.float16, model)
                w = model(extracted_condition=fc, extracted_condition_audio=fca, prompt=prompt, negative_prompt="", num_inference_steps=cfg["denoise_step"], guidance_scale_text=cfg["guidance_scale_text"],
                          guidance_scale_con=cfg["guidance_scale_con"], guidance_scale_audio=cfg["guidance_scale_audio"], num_waveforms_per_prompt=1, audio_end_in_s=2097152 / 44100, generator=torch.Generator().manual_seed(sd)).audios
            g = w[0].float().mean(0).cpu(); g = AF.resample(g[: 44100 * 30], 44100, SR).numpy()[: len(voc)]; g = np.pad(g, (0, len(voc) - len(g)))
            g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, sd, n, flush=True)
            if a.limit_gen and n >= a.limit_gen: raise SystemExit
print("done", n)
