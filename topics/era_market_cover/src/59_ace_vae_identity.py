"""Identity control for the ACE-Step route: the ORIGINAL accompaniment stem is passed through ACE-Step's VAE (encode, decode) without diffusion, then handled exactly like a generation (24 kHz, RMS-matched, remixed with the original vocal),
and written for all four prompts (identical audio) so that 10_eval_gen.py / 13_content_metrics.py / 54_vinet_identity.py can score it as a generation directory (ace_vae_id, seed 0 only). It shows how far the codec-matched era probes
move because of the VAE round trip alone. Run with venv_ace from ~/era_market/ext/ACE-Step-1.5.  usage: python ../../src/59_ace_vae_identity.py"""
import os, sys, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from acestep.handler import AceStepHandler
from common import ROOT, SR
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; out = os.path.join(ROOT, "ace_vae_id"); os.makedirs(out, exist_ok=True); open(os.path.join(out, "picked.txt"), "w").write(open(os.path.join(G, "picked.txt")).read())
dit = AceStepHandler(); print(dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cuda"), flush=True)
for s in ids:
    z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    a48 = AF.resample(torch.from_numpy(acc0), SR, 48000)[None].repeat(2, 1).to("cuda")           # (2, T) stereo
    with torch.inference_mode():
        lat = dit._encode_audio_to_latents(a48)                                                   # (T', 64)
        wav = dit.vae.decode(lat.transpose(0, 1)[None].to(dit.vae.dtype)).sample[0].float().mean(0).cpu()
    g = AF.resample(wav, 48000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
    for era in ("neutral", "1960s", "1980s", "2010s"):
        fp = os.path.join(out, f"{s}_{era}_s0"); sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR)
    print(s, flush=True)
