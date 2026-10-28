"""G07 (entropy matching, HW2 idea): teacher-forced predictive entropy of MusicGen-Melody on EnCodec codes of (a) the REAL accompaniment stem and (b) the generated accompaniment of each sampling configuration.
Same measurement for every source: 24 validation songs, frames 250-750 (5-15 s at 50 Hz), neutral prompt, chroma of the original mix over the same span, no CFG (conditional branch only), bf16, mean over positions and the 4 codebooks of H = -sum p log p (nats) and of the cross-entropy -log p(code) (nats).
Reading: if sampled music is 'less surprising' than real music under the model, its mean entropy is lower than that of real music; a temperature whose generated entropy matches the real-music entropy is the entropy-matched setting.
usage: python 72_entropy_match.py real=gen:real  T1.0_k50=gen  T0.8=gen_sp_t08 ... (label=dir; dir holds *_neutral_s0_acc.wav; 'real' reads the stems)  -> results/entropy_match.json"""
import sys, os, json, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest
ids = [l.split(",")[0] for l in open(os.path.join(ROOT, "gen", "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
cfg = model.config.decoder; K = cfg.num_codebooks; PAD = cfg.pad_token_id; BOS = cfg.bos_token_id; F0, F1 = 250, 750
mix = {s: sf.read(rows[s]["path"], dtype="float32")[0] for s in ids}
def measure(wavs):   # wavs: {song: 24 kHz mono acc}
    H, CE = [], []
    for s in ids:
        x = AF.resample(torch.from_numpy(wavs[s].astype(np.float32)), SR, 32000)[None, None].to("cuda").bfloat16()
        with torch.no_grad(): c = model.audio_encoder.encode(x).audio_codes[0, 0][:, F0:F1].long()   # (4, 500)
        m = AF.resample(torch.from_numpy(mix[s]), SR, 32000).numpy()[int(F0 / 50 * 32000):int(F1 / 50 * 32000)]
        inp = proc(text=["a song recording"], audio=m, sampling_rate=32000, padding=True, return_tensors="pt").to("cuda"); inp = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
        T = c.shape[1]; d = torch.full((1, K, T + K - 1), PAD, device="cuda", dtype=torch.long)
        for k in range(K): d[:, k, k:k + T] = c[k]
        dec_in = torch.cat([torch.full((1, K, 1), BOS, device="cuda", dtype=torch.long), d[:, :, :-1]], 2).reshape(K, -1)
        with torch.no_grad(): out = model(**inp, decoder_input_ids=dec_in)
        lg = out.logits.float().reshape(1, K, -1, out.logits.shape[-1])[:, :, -d.shape[2]:]; lp = torch.log_softmax(lg, -1); valid = d != PAD
        ent = -(lp.exp() * lp).sum(-1); ce = -lp.gather(-1, d.masked_fill(d == PAD, 0)[..., None])[..., 0]
        H.append(ent[valid].mean().item()); CE.append(ce[valid].mean().item())
    return {"entropy_mean": float(np.mean(H)), "entropy_sd_over_songs": float(np.std(H)), "xent_mean": float(np.mean(CE)), "xent_sd_over_songs": float(np.std(CE)), "n_songs": len(ids)}
res = {}
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    if d == "real": W = {s: np.load(os.path.join(ROOT, "stems", "A", s + ".npz"))["accomp"] for s in ids}
    else: W = {s: sf.read(os.path.join(ROOT, d, f"{s}_neutral_s0_acc.wav"), dtype="float32")[0] for s in ids}
    res[label] = measure(W); print(label, res[label], flush=True)
json.dump(res, open(os.path.join(ROOT, "results", "entropy_match.json"), "w"), indent=1)
