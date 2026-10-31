"""Variant 61: three-branch guidance that keeps the standard melody classifier-free guidance and adds the era contrast on top:
logits = L(null) + gm (L(neutral+chroma) - L(null)) + ge (L(era+chroma) - L(neutral+chroma)), computed on log-softmaxed branch logits.
gm = ge gives the standard CFG with the era prompt (L(null) + g (L(era+chroma) - L(null))); ge > gm amplifies the era contrast beyond it while the melody term stays at gm.
Implementation: the batch is duplicated to 2 rows, the encoder states of the 4 CFG rows are [era+chroma, neutral+chroma, null, null]; a logits processor (runs before HF's CFG processor) combines the three branches per codebook, draws ONE token (top-k, T=1, torch RNG seeded per clip) and returns a one-hot over it for both rows so that the rows stay identical.
Same 24 validation songs, original-vocal remix protocol and RMS matching as 07_generate.py; neutral rows copied from NEUTRAL_SRC (default gen); LORA=<adapter> merges a LoRA as in 07_generate.py.
usage: python 61_triple_guidance.py --gm 3 --ge 5 --outdir gen_tri_g3e5 --seeds 0 1 2"""
import argparse, os, shutil, glob, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest
ERAS = {"1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
NEG = {"1960s": "2010s", "1980s": "2010s", "2010s": "1960s"}
ap = argparse.ArgumentParser(); ap.add_argument("--gm", type=float, default=3.0); ap.add_argument("--ge", type=float, default=5.0); ap.add_argument("--outdir", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--eras", nargs="*", default=None, help="decade prompt keys from prompts_era.ERAS (default 1960s 1980s 2010s)"); a = ap.parse_args()
if a.eras:
    import prompts_era; ERAS = {k: prompts_era.ERAS[k] for k in a.eras}
G = os.path.join(ROOT, "gen"); ids = [l.split(",")[0] for l in open(os.path.join(G, "picked.txt")).read().split("\n")]; rows = {r["sample_id"]: r for r in manifest("A")}
out = os.path.join(ROOT, a.outdir); os.makedirs(out, exist_ok=True); shutil.copy(os.path.join(G, "picked.txt"), os.path.join(out, "picked.txt"))
NS = os.path.join(ROOT, os.environ.get("NEUTRAL_SRC", "gen"))   # LoRA runs: NEUTRAL_SRC=gen_lora (neutral rows of the same adapter)
for f in glob.glob(os.path.join(NS, "*_neutral_s*_*.wav")):
    if int(os.path.basename(f).split("_s")[1][0]) in a.seeds: shutil.copy(f, out)
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid); model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval()
if os.environ.get("LORA"):   # adapters from 48_lora_musicgen.py merged into the base weights, as in 07_generate.py
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, "lora", os.environ["LORA"])).merge_and_unload().eval()
NEUTRAL = "a song recording"
STATE = {}; orig = MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation
def patched(self, inputs_tensor, model_kwargs, model_input_name, generation_config):
    nb = STATE.get("neutral"); gs = generation_config.guidance_scale
    if nb is None or gs is None or gs <= 1: return orig(self, inputs_tensor, model_kwargs, model_input_name, generation_config)
    kw_n = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in model_kwargs.items()}; kw_n["attention_mask"] = nb["attention_mask"]
    out_e = orig(self, inputs_tensor, model_kwargs, model_input_name, generation_config); out_n = orig(self, nb["input_ids"], kw_n, model_input_name, generation_config)
    he = out_e["encoder_hidden_states"]; hn = out_n["encoder_hidden_states"]; assert he.shape[0] == 4
    out_e["encoder_hidden_states"] = torch.cat([he[0:1], hn[0:1], he[2:4]], dim=0); return out_e          # rows [era+chroma, neutral+chroma, null, null]
from transformers import LogitsProcessor, LogitsProcessorList
K = 4   # codebooks folded into the decoder batch
class Triple(LogitsProcessor):
    def __call__(self, input_ids, scores):
        assert scores.shape[0] == 4 * K, scores.shape
        lp = torch.log_softmax(scores.float(), dim=-1); cond, unc = lp[:2 * K], lp[2 * K:]
        Le, Ln, L0 = cond[:K], cond[K:], unc[:K]
        C = L0 + a.gm * (Ln - L0) + a.ge * (Le - Ln)
        if a.top_k > 0:
            v = torch.topk(C, a.top_k, dim=-1).values[:, -1:]; C = C.masked_fill(C < v, -1e9)
        tok = torch.multinomial(torch.softmax(C, dim=-1), 1)               # [K,1], one token per codebook
        oh = torch.full_like(Le, -1e9).scatter(1, tok, 0.0)
        oh = torch.cat([oh, oh], dim=0); return torch.cat([oh, oh], dim=0).to(scores.dtype)   # cond and uncond halves identical -> HF CFG leaves it unchanged
MusicgenMelodyForConditionalGeneration._prepare_encoder_hidden_states_kwargs_for_generation = patched
n = 0
for s in ids:
    mix = sf.read(rows[s]["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", "A", s + ".npz")); voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32); m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era, prompt in ERAS.items():
        for seed in a.seeds:
            fp = os.path.join(out, f"{s}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            tok = proc(text=[prompt, NEUTRAL], padding=True, return_tensors="pt"); feat = proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt")["input_features"]
            STATE["neutral"] = {"input_ids": tok["input_ids"][1:2].repeat(2, 1).to("cuda"), "attention_mask": tok["attention_mask"][1:2].repeat(2, 1).to("cuda")}
            inp = {"input_ids": tok["input_ids"][0:1].repeat(2, 1).to("cuda"), "attention_mask": tok["attention_mask"][0:1].repeat(2, 1).to("cuda"), "input_features": feat.repeat(2, 1, 1).to("cuda").half()}
            torch.manual_seed(seed)
            with torch.no_grad(): o = model.generate(**inp, do_sample=True, top_k=a.top_k, guidance_scale=3.0, logits_processor=LogitsProcessorList([Triple()]), max_new_tokens=1500)
            g = AF.resample(o[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]; g = np.pad(g, (0, len(voc) - len(g))); g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1; print(s, era, seed, n, flush=True)
            if a.limit and n >= a.limit: raise SystemExit
print("done", n)
