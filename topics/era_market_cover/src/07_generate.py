"""Era-targeted accompaniment regeneration with MusicGen-Melody (facebook/musicgen-melody, 1.5B, fp16).
for each validation clip of dataset A: chroma of the ORIGINAL MIX (melody + harmony) + text prompt of a target era -> new 30 s audio; it replaces the accompaniment stem only.
remix = original vocal stem + generated accompaniment scaled to the RMS of the original accompaniment.
saves gen/<sample_id>_<era>_s<seed>_{acc,remix}.wav (24 kHz mono).  usage: python 07_generate.py --n 12 --seeds 0 1"""
import argparse, os, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from common import ROOT, SR, manifest, LABELS
ERAS = {"neutral": "a song recording",
        "1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
ERAS_EXTRA = {"1970s": "a 1970s rock and funk recording, warm analog tape, electric piano, wah guitar, live drums, vinyl-era mix",
              "1990s": "a 1990s pop and hip hop recording, sampled breakbeat drums, early digital synthesizers, CD-era production",
              "2000s": "a 2000s pop and r&b recording, programmed drums, bright digital synths, autotuned pop production, loud compressed mastering",
              "1960s_b": "a 1960s Motown soul record, tambourine, electric bass, brass section, mono mix, tape hiss",
              "1960s_c": "an old vintage recording from the sixties"}
MARKETS = {"neutral": "a song recording",
           "US": "an American 1980s pop and rock recording, gated reverb drums, electric guitars, synthesizers, radio-friendly production",
           "UK": "a British 1980s new wave and synth pop recording, drum machine, analog synthesizers, jangly guitar, dry English production",
           "Brazil": "a Brazilian 1980s MPB and samba-pop recording, rich percussion, cavaquinho, acoustic guitar, warm bass, tropical rhythm",
           "Spain": "a Spanish 1980s pop and rock recording, flamenco-influenced guitar, Latin pop rhythm, Mediterranean sound",
           "Germany": "a German 1980s Neue Deutsche Welle and electronic recording, cold synthesizers, sequencers, drum machine, Berlin sound",
           "Italy": "an Italian 1980s pop and italo disco recording, melodic synthesizers, disco bass, string pads, Italian pop production"}
ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="A", choices=["A", "B"], help="B: market-targeted prompts (MARKETS) on dataset-B validation clips"); ap.add_argument("--eras", nargs="*", default=None, help="subset of prompt keys (ERAS + ERAS_EXTRA); default the four main prompts"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--limit_gen", type=int, default=0)
ap.add_argument("--guidance", type=float, default=3.0); ap.add_argument("--no_melody", action="store_true", help="text-only baseline: do not pass the melody (chroma) condition"); ap.add_argument("--reverse", action="store_true", help="process songs in reverse order (to run a second worker)"); a = ap.parse_args()
ERAS.update(ERAS_EXTRA)
if a.ds == "B": ERAS = dict(MARKETS)
if a.eras: ERAS = {k: ERAS[k] for k in a.eras}
rows = [r for r in manifest(a.ds) if r["split"] == "validation"]
per = a.n // 6; pick = []
for lab in LABELS[a.ds]: pick += [r for r in rows if r["label"] == lab][:per]       # per decade, first ones in manifest order
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid)
model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).to("cuda").eval()
if os.environ.get("LORA"):   # fine-tuned adapters from 48_lora_musicgen.py, merged into the base weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, "lora", os.environ["LORA"])).merge_and_unload().eval()
os.makedirs(os.path.join(ROOT, os.environ.get("GENDIR", "gen")), exist_ok=True); n = 0
with open(os.path.join(ROOT, os.environ.get("GENDIR", "gen"), "picked.txt"), "w") as f: f.write("\n".join(r["sample_id"] + "," + r["label"] for r in pick))
for r in (pick[::-1] if a.reverse else pick):
    mix = sf.read(r["path"], dtype="float32")[0]; z = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz"))
    voc, acc0 = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    m32 = AF.resample(torch.from_numpy(mix), SR, 32000).numpy()
    for era, prompt in ERAS.items():
        for seed in a.seeds:
            fp = os.path.join(ROOT, os.environ.get("GENDIR", "gen"), f"{r['sample_id']}_{era}_s{seed}")
            if os.path.exists(fp + "_remix.wav"): continue
            inp = (proc(text=[prompt], padding=True, return_tensors="pt") if a.no_melody else proc(audio=m32, sampling_rate=32000, text=[prompt], padding=True, return_tensors="pt")).to("cuda")
            inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
            torch.manual_seed(seed)
            with torch.no_grad(): out = model.generate(**inp, do_sample=True, guidance_scale=a.guidance, max_new_tokens=1500)
            g = AF.resample(out[0, 0].float().cpu(), 32000, SR).numpy()[:len(voc)]
            g = np.pad(g, (0, len(voc) - len(g)))
            g = g * (np.sqrt((acc0 ** 2).mean()) / (np.sqrt((g ** 2).mean()) + 1e-9))
            sf.write(fp + "_acc.wav", g, SR); sf.write(fp + "_remix.wav", np.clip(voc + g, -1, 1), SR); n += 1
            print(r["sample_id"], r["label"], era, seed, n, flush=True)
            if a.limit_gen and n >= a.limit_gen: raise SystemExit
print("done", n)
