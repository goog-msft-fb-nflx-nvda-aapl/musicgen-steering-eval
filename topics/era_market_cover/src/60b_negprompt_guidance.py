"""Variant 60b: the contrast branch is a NEGATIVE era prompt (opposite-era description) instead of the neutral prompt: logits = L(neg+chroma) + g (L(era+chroma) - L(neg+chroma)). Negative-prompt guidance as in text-to-image CFG; mapping 1960s->2010s, 1980s->2010s, 2010s->1960s (choice made before any result). Otherwise identical to 60_delta_guidance.py.

Era-contrast ("delta") classifier-free guidance for MusicGen-Melody, after the composable / relative guidance idea (Liu et al. 2022, cited in the course HW3 slides, not re-read here).
Standard CFG: logits = L(null) + g (L(text_era + chroma) - L(null)). Here the null branch is replaced by the NEUTRAL prompt branch WITH the same melody chroma:
logits = L(neutral + chroma) + g (L(era + chroma) - L(neutral + chroma)),  g > 1 amplifies only the era-versus-neutral difference and leaves the melody conditioning in both branches.
Implemented by wrapping MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation: the second (null) half of the encoder states is replaced by the conditional half computed for the neutral prompt.
Same 24 validation songs, original-vocal remix protocol and RMS matching as 07_generate.py; neutral rows are copied from gen/ (standard g=3 neutral) so that 10_eval_gen.py can score the directory.
usage: python 60b_negprompt_guidance.py --g 3 --outdir gen_delta_g3 --seeds 0 1 2"""
import argparse, os, shutil, glob, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest
ERAS = {"1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
NEG = {"1960s": "2010s", "1980s": "2010s", "2010s": "1960s"}
ap = argparse.ArgumentParser(); ap.add_argument("--g", type=float, default=3.0); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--top_k", type=int, default=50); a = ap.parse_args()
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(G, "picked.txt"), os.path.join(out, "picked.txt"))
for f in glob.glob(os.path.join(G, "*_neutral_s*_*.wav")):
    if int(os.path.basename(f).split("_s")[1][0]) in a.seeds: shutil.copy(f, out)
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval()
STATE = {}; orig = MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation
def patched(self, inputs_tensor, model_kwargs, model_input_name, generation_config):
    nb = STATE.get("neutral"); gs = generation_config.guidance_scale
    if nb is None or gs is None or gs <= 1: return orig(self, inputs_tensor, model_kwargs, model_input_name, generation_config)
    kw_n = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in model_kwargs.items()}; kw_n["attention_mask"] = nb["attention_mask"]
    out_e = orig(self, inputs_tensor, model_kwargs, model_input_name, generation_config); out_n = orig(self, nb["input_ids"], kw_n, model_input_name, generation_config)
    B = out_e["encoder_hidden_states"].shape[0] // 2; he = out_e["encoder_hidden_states"]; hn = out_n["encoder_hidden_states"]
    out_e["encoder_hidden_states"] = torch.cat([he[:B], hn[:B]], dim=0); return out_e          # [era+chroma ; neutral+chroma]
MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation = patched
n = 0
for s in ids:
    mix = sf.read(rows[s]["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era, prompt in ERAS.items():
        for seed in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            tok = proc(text=[prompt, ERAS[NEG[era]]], padding=True, return_tensors="pt"); feat = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt")["input_features"]
            STATE["neutral"] = {"input_ids": tok["input_ids"][1:2].to("cuda"), "attention_mask": tok["attention_mask"][1:2].to("cuda")}
            inp = {"input_ids": tok["input_ids"][0:1].to("cuda"), "attention_mask": tok["attention_mask"][0:1].to("cuda"), "input_features": feat.to("cuda").half()}
            torch.manual_seed(seed)
            with torch.no_grad(): o = model.generate(**inp, do_sample=True, top_k=a.top_k, guidance_scale=a.g, max_new_tokens=1500)
            g = AF.resample(o[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, seed, n, flush=True)
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
